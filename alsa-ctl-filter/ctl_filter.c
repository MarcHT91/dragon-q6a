/*
 * ctl_filter - ALSA external control plugin that exposes only a whitelist of
 * controls from a hardware card, optionally renamed and with a clamped range.
 *
 * Use case: Qualcomm ASoC cards (e.g. Radxa Dragon Q6A / QCS6490) expose ~200
 * routing/codec controls; alsamixer becomes unusable. Point ctl.!default at
 * this plugin and alsamixer shows only the volumes you care about, while
 * `alsamixer -D sysdefault` still shows the full card.
 *
 * Config (~/.asoundrc):
 *   ctl_type.filter { lib "/path/libasound_module_ctl_filter.so" }
 *   ctl.!default {
 *     type filter
 *     card QCS6490RadxaDra          # card name or index, default 0
 *     # slave "sysdefault:0"        # child ctl, default sysdefault:<card>
 *     controls [
 *       { name "Headphone Playback Volume" source "HP Volume" }
 *       { name "HDMI Playback Volume"
 *         source "stream1.vol_ctrl1 MultiMedia2 Playback Volu"
 *         max 8192                   # clamp range (here: 8192 = unity gain)
 *         linear_db true }           # advertise linear dB scale (-inf..0 dB)
 *     ]
 *   }
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <alsa/asoundlib.h>
#include <alsa/control_external.h>

#define MAXCTL 32

struct fctl {
	char name[44];
	snd_ctl_elem_id_t *hwid;
	snd_ctl_elem_type_t type;
	unsigned int count, access;
	long min, max, step;
	unsigned int items;
	int linear_db;
	unsigned int tlv[64];
	int tlv_len;	/* bytes, 0 = none */
};

struct filter {
	snd_ctl_ext_t ext;
	snd_ctl_t *hw;
	int nctl;
	struct fctl c[MAXCTL];
};

static void f_close(snd_ctl_ext_t *ext)
{
	struct filter *f = ext->private_data;
	for (int i = 0; i < f->nctl; i++)
		snd_ctl_elem_id_free(f->c[i].hwid);
	snd_ctl_close(f->hw);
	free(f);
}

static int f_elem_count(snd_ctl_ext_t *ext)
{
	return ((struct filter *)ext->private_data)->nctl;
}

static int f_elem_list(snd_ctl_ext_t *ext, unsigned int off, snd_ctl_elem_id_t *id)
{
	struct filter *f = ext->private_data;
	if (off >= (unsigned)f->nctl)
		return -EINVAL;
	snd_ctl_elem_id_set_interface(id, SND_CTL_ELEM_IFACE_MIXER);
	snd_ctl_elem_id_set_name(id, f->c[off].name);
	return 0;
}

static snd_ctl_ext_key_t f_find_elem(snd_ctl_ext_t *ext, const snd_ctl_elem_id_t *id)
{
	struct filter *f = ext->private_data;
	unsigned int numid = snd_ctl_elem_id_get_numid(id);
	if (numid > 0 && numid <= (unsigned)f->nctl)
		return numid - 1;
	const char *name = snd_ctl_elem_id_get_name(id);
	for (int i = 0; i < f->nctl; i++)
		if (!strcmp(name, f->c[i].name))
			return i;
	return SND_CTL_EXT_KEY_NOT_FOUND;
}

static int f_get_attribute(snd_ctl_ext_t *ext, snd_ctl_ext_key_t key,
			   int *type, unsigned int *acc, unsigned int *count)
{
	struct fctl *c = &((struct filter *)ext->private_data)->c[key];
	*type = c->type;
	*count = c->count;
	*acc = c->access;
	return 0;
}

static int f_get_integer_info(snd_ctl_ext_t *ext, snd_ctl_ext_key_t key,
			      long *imin, long *imax, long *istep)
{
	struct fctl *c = &((struct filter *)ext->private_data)->c[key];
	*imin = c->min;
	*imax = c->max;
	*istep = c->step;
	return 0;
}

static int f_get_enumerated_info(snd_ctl_ext_t *ext, snd_ctl_ext_key_t key, unsigned int *items)
{
	*items = ((struct filter *)ext->private_data)->c[key].items;
	return 0;
}

static int f_get_enumerated_name(snd_ctl_ext_t *ext, snd_ctl_ext_key_t key,
				 unsigned int item, char *name, size_t len)
{
	struct filter *f = ext->private_data;
	snd_ctl_elem_info_t *info;
	int err;
	snd_ctl_elem_info_alloca(&info);
	snd_ctl_elem_info_set_id(info, f->c[key].hwid);
	snd_ctl_elem_info_set_item(info, item);
	if ((err = snd_ctl_elem_info(f->hw, info)) < 0)
		return err;
	snprintf(name, len, "%s", snd_ctl_elem_info_get_item_name(info));
	return 0;
}

static int hw_read(struct filter *f, snd_ctl_ext_key_t key, snd_ctl_elem_value_t *v)
{
	snd_ctl_elem_value_set_id(v, f->c[key].hwid);
	return snd_ctl_elem_read(f->hw, v);
}

static int f_read_integer(snd_ctl_ext_t *ext, snd_ctl_ext_key_t key, long *value)
{
	struct filter *f = ext->private_data;
	struct fctl *c = &f->c[key];
	snd_ctl_elem_value_t *v;
	int err;
	snd_ctl_elem_value_alloca(&v);
	if ((err = hw_read(f, key, v)) < 0)
		return err;
	for (unsigned i = 0; i < c->count; i++) {
		long x = c->type == SND_CTL_ELEM_TYPE_BOOLEAN ?
			snd_ctl_elem_value_get_boolean(v, i) :
			snd_ctl_elem_value_get_integer(v, i);
		value[i] = x > c->max ? c->max : x;	/* clamped view */
	}
	return 0;
}

static int f_read_enumerated(snd_ctl_ext_t *ext, snd_ctl_ext_key_t key, unsigned int *items)
{
	struct filter *f = ext->private_data;
	snd_ctl_elem_value_t *v;
	int err;
	snd_ctl_elem_value_alloca(&v);
	if ((err = hw_read(f, key, v)) < 0)
		return err;
	for (unsigned i = 0; i < f->c[key].count; i++)
		items[i] = snd_ctl_elem_value_get_enumerated(v, i);
	return 0;
}

static int f_write_integer(snd_ctl_ext_t *ext, snd_ctl_ext_key_t key, long *value)
{
	struct filter *f = ext->private_data;
	struct fctl *c = &f->c[key];
	snd_ctl_elem_value_t *v;
	int err, changed = 0;
	snd_ctl_elem_value_alloca(&v);
	if ((err = hw_read(f, key, v)) < 0)
		return err;
	for (unsigned i = 0; i < c->count; i++) {
		long x = value[i] < c->min ? c->min : value[i] > c->max ? c->max : value[i];
		if (c->type == SND_CTL_ELEM_TYPE_BOOLEAN) {
			changed |= snd_ctl_elem_value_get_boolean(v, i) != x;
			snd_ctl_elem_value_set_boolean(v, i, x);
		} else {
			changed |= snd_ctl_elem_value_get_integer(v, i) != x;
			snd_ctl_elem_value_set_integer(v, i, x);
		}
	}
	if (!changed)
		return 0;
	return (err = snd_ctl_elem_write(f->hw, v)) < 0 ? err : 1;
}

static int f_write_enumerated(snd_ctl_ext_t *ext, snd_ctl_ext_key_t key, unsigned int *items)
{
	struct filter *f = ext->private_data;
	snd_ctl_elem_value_t *v;
	int err;
	snd_ctl_elem_value_alloca(&v);
	if ((err = hw_read(f, key, v)) < 0)
		return err;
	for (unsigned i = 0; i < f->c[key].count; i++)
		snd_ctl_elem_value_set_enumerated(v, i, items[i]);
	return (err = snd_ctl_elem_write(f->hw, v)) < 0 ? err : 1;
}

static int f_tlv(snd_ctl_ext_t *ext, snd_ctl_ext_key_t key, int op_flag,
		 unsigned int numid, unsigned int *tlv, unsigned int tlv_size)
{
	struct fctl *c = &((struct filter *)ext->private_data)->c[key];
	(void)numid;
	if (op_flag != 0)	/* read only */
		return -ENXIO;
	if (!c->tlv_len)
		return -ENXIO;
	if (tlv_size < (unsigned)c->tlv_len)
		return -ENOMEM;
	memcpy(tlv, c->tlv, c->tlv_len);
	return c->tlv_len;
}

static void f_subscribe_events(snd_ctl_ext_t *ext, int subscribe)
{
	struct filter *f = ext->private_data;
	snd_ctl_subscribe_events(f->hw, subscribe);
}

static int f_read_event(snd_ctl_ext_t *ext, snd_ctl_elem_id_t *id, unsigned int *event_mask)
{
	struct filter *f = ext->private_data;
	snd_ctl_event_t *ev;
	snd_ctl_event_alloca(&ev);
	for (;;) {
		int err = snd_ctl_read(f->hw, ev);
		if (err <= 0)
			return err == 0 ? -EAGAIN : err;
		if (snd_ctl_event_get_type(ev) != SND_CTL_EVENT_ELEM)
			continue;
		unsigned int numid = snd_ctl_event_elem_get_numid(ev);
		for (int i = 0; i < f->nctl; i++) {
			if (snd_ctl_elem_id_get_numid(f->c[i].hwid) != numid)
				continue;
			snd_ctl_elem_id_clear(id);
			snd_ctl_elem_id_set_numid(id, i + 1);
			snd_ctl_elem_id_set_interface(id, SND_CTL_ELEM_IFACE_MIXER);
			snd_ctl_elem_id_set_name(id, f->c[i].name);
			*event_mask = snd_ctl_event_elem_get_mask(ev) & ~SND_CTL_EVENT_MASK_ADD;
			if (!*event_mask)
				break;
			return 1;
		}
	}
}

static const snd_ctl_ext_callback_t filter_cb = {
	.close = f_close,
	.elem_count = f_elem_count,
	.elem_list = f_elem_list,
	.find_elem = f_find_elem,
	.get_attribute = f_get_attribute,
	.get_integer_info = f_get_integer_info,
	.get_enumerated_info = f_get_enumerated_info,
	.get_enumerated_name = f_get_enumerated_name,
	.read_integer = f_read_integer,
	.read_enumerated = f_read_enumerated,
	.write_integer = f_write_integer,
	.write_enumerated = f_write_enumerated,
	.subscribe_events = f_subscribe_events,
	.read_event = f_read_event,
};

/* Look up the hardware control `src` and fill in c. */
static int setup_ctl(struct filter *f, struct fctl *c, const char *src, long max_override)
{
	snd_ctl_elem_info_t *info;
	int err;
	snd_ctl_elem_info_alloca(&info);
	snd_ctl_elem_id_malloc(&c->hwid);
	snd_ctl_elem_id_set_interface(c->hwid, SND_CTL_ELEM_IFACE_MIXER);
	snd_ctl_elem_id_set_name(c->hwid, src);
	snd_ctl_elem_info_set_id(info, c->hwid);
	if ((err = snd_ctl_elem_info(f->hw, info)) < 0) {
		SNDERR("filter: control '%s' not found", src);
		return err;
	}
	snd_ctl_elem_info_get_id(info, c->hwid);	/* now has numid */
	c->type = snd_ctl_elem_info_get_type(info);
	c->count = snd_ctl_elem_info_get_count(info);
	c->access = SND_CTL_EXT_ACCESS_READWRITE;
	switch (c->type) {
	case SND_CTL_ELEM_TYPE_BOOLEAN:
		c->min = 0; c->max = 1; c->step = 0;
		break;
	case SND_CTL_ELEM_TYPE_INTEGER:
		c->min = snd_ctl_elem_info_get_min(info);
		c->max = snd_ctl_elem_info_get_max(info);
		c->step = snd_ctl_elem_info_get_step(info);
		if (max_override >= 0 && max_override < c->max)
			c->max = max_override;
		break;
	case SND_CTL_ELEM_TYPE_ENUMERATED:
		c->items = snd_ctl_elem_info_get_items(info);
		break;
	default:
		SNDERR("filter: control '%s' has unsupported type", src);
		return -EINVAL;
	}
	if (c->type != SND_CTL_ELEM_TYPE_INTEGER)
		return 0;
	if (c->linear_db) {
		/* linear gain: min -> mute, max -> 0 dB */
		c->tlv[0] = SND_CTL_TLVT_DB_LINEAR;
		c->tlv[1] = 2 * sizeof(int);
		c->tlv[2] = (unsigned int)SND_CTL_TLV_DB_GAIN_MUTE;
		c->tlv[3] = 0;
		c->tlv_len = 4 * sizeof(int);
	} else if (snd_ctl_elem_info_is_tlv_readable(info) &&
		   snd_ctl_elem_tlv_read(f->hw, c->hwid, c->tlv, sizeof(c->tlv)) >= 0) {
		c->tlv_len = c->tlv[1] + 2 * sizeof(int);
	}
	if (c->tlv_len)
		c->access |= SND_CTL_EXT_ACCESS_TLV_READ | SND_CTL_EXT_ACCESS_TLV_CALLBACK;
	return 0;
}

SND_CTL_PLUGIN_DEFINE_FUNC(filter)
{
	snd_config_iterator_t i, next;
	snd_config_t *controls = NULL;
	const char *card = "0", *slave = NULL;
	char hwname[64];
	struct filter *f;
	int err;
	(void)root;

	snd_config_for_each(i, next, conf) {
		snd_config_t *n = snd_config_iterator_entry(i);
		const char *id;
		if (snd_config_get_id(n, &id) < 0)
			continue;
		if (!strcmp(id, "comment") || !strcmp(id, "type") || !strcmp(id, "hint"))
			continue;
		if (!strcmp(id, "card")) {
			long idx;
			if (snd_config_get_integer(n, &idx) == 0) {
				static char buf[16];
				snprintf(buf, sizeof(buf), "%ld", idx);
				card = buf;
			} else if (snd_config_get_string(n, &card) < 0) {
				SNDERR("filter: invalid card");
				return -EINVAL;
			}
			continue;
		}
		if (!strcmp(id, "slave")) {
			if (snd_config_get_string(n, &slave) < 0) {
				SNDERR("filter: invalid slave");
				return -EINVAL;
			}
			continue;
		}
		if (!strcmp(id, "controls")) {
			controls = n;
			continue;
		}
		SNDERR("filter: unknown field %s", id);
		return -EINVAL;
	}
	if (!controls) {
		SNDERR("filter: no controls defined");
		return -EINVAL;
	}

	f = calloc(1, sizeof(*f));
	if (!f)
		return -ENOMEM;
	/* sysdefault: includes UCM LibraryConfig remaps (e.g. "HP Volume") */
	if (slave)
		snprintf(hwname, sizeof(hwname), "%s", slave);
	else
		snprintf(hwname, sizeof(hwname), "sysdefault:%s", card);
	if ((err = snd_ctl_open(&f->hw, hwname, SND_CTL_NONBLOCK)) < 0) {
		free(f);
		return err;
	}

	snd_config_for_each(i, next, controls) {
		snd_config_t *n = snd_config_iterator_entry(i), *e;
		const char *name = NULL, *src = NULL;
		long max = -1;
		if (f->nctl >= MAXCTL)
			break;
		struct fctl *c = &f->c[f->nctl];
		if (snd_config_search(n, "source", &e) == 0)
			snd_config_get_string(e, &src);
		if (snd_config_search(n, "name", &e) == 0)
			snd_config_get_string(e, &name);
		if (snd_config_search(n, "max", &e) == 0)
			snd_config_get_integer(e, &max);
		if (snd_config_search(n, "linear_db", &e) == 0)
			c->linear_db = snd_config_get_bool(e) > 0;
		if (!src) {
			SNDERR("filter: control entry without source");
			err = -EINVAL;
			goto fail;
		}
		snprintf(c->name, sizeof(c->name), "%s", name ? name : src);
		if ((err = setup_ctl(f, c, src, max)) < 0) {
			snd_ctl_elem_id_free(c->hwid);
			goto fail;
		}
		f->nctl++;
	}

	{
		snd_ctl_card_info_t *ci;
		struct pollfd pfd;
		snd_ctl_card_info_alloca(&ci);
		snd_ctl_card_info(f->hw, ci);
		f->ext.version = SND_CTL_EXT_VERSION;
		f->ext.card_idx = snd_ctl_card_info_get_card(ci);
		snprintf(f->ext.id, sizeof(f->ext.id), "%s", snd_ctl_card_info_get_id(ci));
		snprintf(f->ext.driver, sizeof(f->ext.driver), "%s", snd_ctl_card_info_get_driver(ci));
		snprintf(f->ext.name, sizeof(f->ext.name), "%s", snd_ctl_card_info_get_name(ci));
		snprintf(f->ext.longname, sizeof(f->ext.longname), "%s", snd_ctl_card_info_get_longname(ci));
		snprintf(f->ext.mixername, sizeof(f->ext.mixername), "%s (filtered)",
			 snd_ctl_card_info_get_mixername(ci));
		snd_ctl_poll_descriptors(f->hw, &pfd, 1);
		f->ext.poll_fd = pfd.fd;
	}
	f->ext.callback = &filter_cb;
	f->ext.private_data = f;
	f->ext.tlv.c = f_tlv;

	if ((err = snd_ctl_ext_create(&f->ext, name, mode)) < 0)
		goto fail;
	*handlep = f->ext.handle;
	return 0;

fail:
	for (int k = 0; k < f->nctl; k++)
		snd_ctl_elem_id_free(f->c[k].hwid);
	snd_ctl_close(f->hw);
	free(f);
	return err;
}

SND_CTL_PLUGIN_SYMBOL(filter);
