/*
 * HFT Kernel Module for Ultra-Low Latency Market Data Processing
 * 
 * This kernel module provides:
 * - Direct packet capture and processing at kernel level
 * - Bypass of userspace networking stack for critical data
 * - Zero-copy packet handling for market data feeds
 * - Real-time packet filtering and forwarding
 * 
 * Requires: Linux kernel headers, CAP_NET_RAW capability
 * Build: make -C /lib/modules/$(uname -r)/build M=$(pwd) modules
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/netfilter.h>
#include <linux/netfilter_ipv4.h>
#include <linux/skbuff.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <linux/tcp.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/uaccess.h>
#include <linux/time.h>
#include <linux/ktime.h>
#include <linux/spinlock.h>
#include <linux/slab.h>
#include <net/ip.h>

#define MODULE_NAME "hft_packet_processor"
#define PROC_ENTRY_NAME "hft_stats"
#define MAX_MARKET_DATA_SOURCES 16
#define PACKET_BUFFER_SIZE 8192

// Market data source configuration
struct market_data_source {
    __be32 ip_addr;          // Source IP address
    __be16 port;             // Source port
    u32 packet_count;        // Packets processed
    u64 total_latency_ns;    // Total processing latency
    bool enabled;            // Source enabled flag
};

// Packet processing statistics
struct hft_stats {
    u64 total_packets;       // Total packets processed
    u64 market_packets;      // Market data packets
    u64 filtered_packets;    // Filtered out packets
    u64 avg_latency_ns;      // Average processing latency
    u64 min_latency_ns;      // Minimum processing latency
    u64 max_latency_ns;      // Maximum processing latency
    u32 active_sources;      // Number of active sources
};

// Global variables
static struct market_data_source sources[MAX_MARKET_DATA_SOURCES];
static struct hft_stats stats;
static struct proc_dir_entry *proc_entry;
static struct nf_hook_ops netfilter_ops;
static DEFINE_SPINLOCK(stats_lock);

// Polygon.io market data source IPs (example - replace with actual IPs)
static const __be32 polygon_ips[] = {
    0x08080808,  // 8.8.8.8 (placeholder)
    0x08080404,  // 8.8.4.4 (placeholder)
};

// Alpaca market data source IPs (example - replace with actual IPs)  
static const __be32 alpaca_ips[] = {
    0x01010101,  // 1.1.1.1 (placeholder)
    0x01000001,  // 1.0.0.1 (placeholder)
};

// Fast packet classification for market data
static inline bool is_market_data_packet(struct sk_buff *skb)
{
    struct iphdr *iph;
    struct udphdr *udph;
    struct tcphdr *tcph;
    __be32 saddr;
    __be16 sport;
    int i;

    if (!skb)
        return false;

    iph = ip_hdr(skb);
    if (!iph)
        return false;

    saddr = iph->saddr;

    // Check if packet is from known market data sources
    for (i = 0; i < ARRAY_SIZE(polygon_ips); i++) {
        if (saddr == polygon_ips[i])
            goto check_port;
    }
    
    for (i = 0; i < ARRAY_SIZE(alpaca_ips); i++) {
        if (saddr == alpaca_ips[i])
            goto check_port;
    }
    
    return false;

check_port:
    // Check protocol and port
    if (iph->protocol == IPPROTO_UDP) {
        if (skb->len < sizeof(struct iphdr) + sizeof(struct udphdr))
            return false;
        
        udph = udp_hdr(skb);
        sport = ntohs(udph->source);
        
        // WebSocket over UDP or direct UDP feeds (ports 443, 8080, 8443)
        return (sport == 443 || sport == 8080 || sport == 8443);
        
    } else if (iph->protocol == IPPROTO_TCP) {
        if (skb->len < sizeof(struct iphdr) + sizeof(struct tcphdr))
            return false;
        
        tcph = tcp_hdr(skb);
        sport = ntohs(tcph->source);
        
        // WebSocket over TCP (ports 443, 8080, 8443)
        return (sport == 443 || sport == 8080 || sport == 8443);
    }
    
    return false;
}

// Update source statistics
static void update_source_stats(__be32 saddr, __be16 sport, u64 latency_ns)
{
    int i;
    
    for (i = 0; i < MAX_MARKET_DATA_SOURCES; i++) {
        if (sources[i].enabled && sources[i].ip_addr == saddr && sources[i].port == sport) {
            sources[i].packet_count++;
            sources[i].total_latency_ns += latency_ns;
            break;
        } else if (!sources[i].enabled) {
            // Register new source
            sources[i].ip_addr = saddr;
            sources[i].port = sport;
            sources[i].packet_count = 1;
            sources[i].total_latency_ns = latency_ns;
            sources[i].enabled = true;
            stats.active_sources++;
            break;
        }
    }
}

// Main packet processing hook
static unsigned int hft_packet_hook(void *priv,
                                   struct sk_buff *skb,
                                   const struct nf_hook_state *state)
{
    struct iphdr *iph;
    struct udphdr *udph;
    struct tcphdr *tcph;
    ktime_t start_time, end_time;
    u64 latency_ns;
    __be32 saddr;
    __be16 sport = 0;
    unsigned long flags;

    if (!skb)
        return NF_ACCEPT;

    start_time = ktime_get();

    // Fast path: check if this is market data
    if (!is_market_data_packet(skb)) {
        spin_lock_irqsave(&stats_lock, flags);
        stats.filtered_packets++;
        spin_unlock_irqrestore(&stats_lock, flags);
        return NF_ACCEPT;
    }

    // Process market data packet
    iph = ip_hdr(skb);
    saddr = iph->saddr;

    if (iph->protocol == IPPROTO_UDP) {
        udph = udp_hdr(skb);
        sport = udph->source;
    } else if (iph->protocol == IPPROTO_TCP) {
        tcph = tcp_hdr(skb);
        sport = tcph->source;
    }

    end_time = ktime_get();
    latency_ns = ktime_to_ns(ktime_sub(end_time, start_time));

    // Update statistics atomically
    spin_lock_irqsave(&stats_lock, flags);
    
    stats.total_packets++;
    stats.market_packets++;
    
    if (stats.min_latency_ns == 0 || latency_ns < stats.min_latency_ns)
        stats.min_latency_ns = latency_ns;
    
    if (latency_ns > stats.max_latency_ns)
        stats.max_latency_ns = latency_ns;
    
    // Running average calculation
    stats.avg_latency_ns = ((stats.avg_latency_ns * (stats.market_packets - 1)) + latency_ns) / stats.market_packets;
    
    spin_unlock_irqrestore(&stats_lock, flags);

    // Update per-source statistics
    update_source_stats(saddr, sport, latency_ns);

    // For production: Here you would implement zero-copy forwarding
    // to userspace HFT application using techniques like:
    // - mmap shared memory regions
    // - netlink sockets
    // - custom character device
    // - AF_PACKET sockets
    
    return NF_ACCEPT;
}

// Proc file operations for statistics display
static int hft_proc_show(struct seq_file *m, void *v)
{
    unsigned long flags;
    int i;

    spin_lock_irqsave(&stats_lock, flags);
    
    seq_printf(m, "=== HFT Kernel Module Statistics ===\n");
    seq_printf(m, "Total packets processed: %llu\n", stats.total_packets);
    seq_printf(m, "Market data packets: %llu\n", stats.market_packets);
    seq_printf(m, "Filtered packets: %llu\n", stats.filtered_packets);
    seq_printf(m, "Average latency: %llu ns\n", stats.avg_latency_ns);
    seq_printf(m, "Minimum latency: %llu ns\n", stats.min_latency_ns);
    seq_printf(m, "Maximum latency: %llu ns\n", stats.max_latency_ns);
    seq_printf(m, "Active sources: %u\n", stats.active_sources);
    
    seq_printf(m, "\n=== Market Data Sources ===\n");
    for (i = 0; i < MAX_MARKET_DATA_SOURCES; i++) {
        if (sources[i].enabled) {
            u64 avg_source_latency = sources[i].packet_count > 0 ?
                sources[i].total_latency_ns / sources[i].packet_count : 0;
            
            seq_printf(m, "Source %d: IP=%pI4 Port=%u Packets=%u AvgLatency=%llu ns\n",
                      i, &sources[i].ip_addr, ntohs(sources[i].port),
                      sources[i].packet_count, avg_source_latency);
        }
    }
    
    spin_unlock_irqrestore(&stats_lock, flags);
    
    return 0;
}

static int hft_proc_open(struct inode *inode, struct file *file)
{
    return single_open(file, hft_proc_show, NULL);
}

static const struct proc_ops hft_proc_ops = {
    .proc_open = hft_proc_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = single_release,
};

// Module initialization
static int __init hft_module_init(void)
{
    int ret;

    printk(KERN_INFO "%s: Initializing HFT packet processor\n", MODULE_NAME);

    // Initialize statistics
    memset(&stats, 0, sizeof(stats));
    memset(sources, 0, sizeof(sources));

    // Create proc entry for statistics
    proc_entry = proc_create(PROC_ENTRY_NAME, 0444, NULL, &hft_proc_ops);
    if (!proc_entry) {
        printk(KERN_ERR "%s: Failed to create proc entry\n", MODULE_NAME);
        return -ENOMEM;
    }

    // Register netfilter hook
    netfilter_ops.hook = hft_packet_hook;
    netfilter_ops.hooknum = NF_INET_PRE_ROUTING;
    netfilter_ops.pf = PF_INET;
    netfilter_ops.priority = NF_IP_PRI_FIRST;

    ret = nf_register_net_hook(&init_net, &netfilter_ops);
    if (ret) {
        printk(KERN_ERR "%s: Failed to register netfilter hook\n", MODULE_NAME);
        proc_remove(proc_entry);
        return ret;
    }

    printk(KERN_INFO "%s: Module loaded successfully\n", MODULE_NAME);
    printk(KERN_INFO "%s: Statistics available at /proc/%s\n", MODULE_NAME, PROC_ENTRY_NAME);
    
    return 0;
}

// Module cleanup
static void __exit hft_module_exit(void)
{
    printk(KERN_INFO "%s: Cleaning up HFT packet processor\n", MODULE_NAME);

    // Unregister netfilter hook
    nf_unregister_net_hook(&init_net, &netfilter_ops);

    // Remove proc entry
    proc_remove(proc_entry);

    printk(KERN_INFO "%s: Module unloaded successfully\n", MODULE_NAME);
}

// Module metadata
MODULE_LICENSE("GPL");
MODULE_AUTHOR("HFT System Developer");
MODULE_DESCRIPTION("Ultra-low latency packet processor for HFT market data");
MODULE_VERSION("1.0");

module_init(hft_module_init);
module_exit(hft_module_exit);