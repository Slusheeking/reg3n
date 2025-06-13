#include <linux/module.h>
#define INCLUDE_VERMAGIC
#include <linux/build-salt.h>
#include <linux/elfnote-lto.h>
#include <linux/export-internal.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

#ifdef CONFIG_UNWINDER_ORC
#include <asm/orc_header.h>
ORC_HEADER;
#endif

BUILD_SALT;
BUILD_LTO_INFO;

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(".gnu.linkonce.this_module") = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef CONFIG_RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif



static const struct modversion_info ____versions[]
__used __section("__versions") = {
	{ 0x5b8239ca, "__x86_return_thunk" },
	{ 0x6a354d52, "proc_remove" },
	{ 0x3ca1d81f, "single_open" },
	{ 0xc49519d0, "nf_unregister_net_hook" },
	{ 0x34db050b, "_raw_spin_lock_irqsave" },
	{ 0x2f9b7e16, "seq_printf" },
	{ 0xd35cce70, "_raw_spin_unlock_irqrestore" },
	{ 0x87a21cb3, "__ubsan_handle_out_of_bounds" },
	{ 0x54b1fac6, "__ubsan_handle_load_invalid_value" },
	{ 0xb43f9365, "ktime_get" },
	{ 0x79ccc597, "seq_read" },
	{ 0x35a231ea, "seq_lseek" },
	{ 0xf3c9b210, "single_release" },
	{ 0xbdfb6dbb, "__fentry__" },
	{ 0x122c3a7e, "_printk" },
	{ 0x4f4ab2c1, "proc_create" },
	{ 0x26d7d0e6, "init_net" },
	{ 0x399b74ea, "nf_register_net_hook" },
	{ 0x6ad2b3e, "module_layout" },
};

MODULE_INFO(depends, "");


MODULE_INFO(srcversion, "36D6D87C4F804EDD4A4EAB4");
