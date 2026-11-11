PREFIX ?= $(HOME)/.local/lib/alsa-lib
CFLAGS ?= -O2 -Wall -Wextra
SO = libasound_module_ctl_filter.so

$(SO): ctl_filter.c
	$(CC) $(CFLAGS) -DPIC -fPIC -shared -o $@ $< -lasound

install: $(SO)
	install -D -m 0755 $(SO) $(PREFIX)/$(SO)

clean:
	rm -f $(SO)
.PHONY: install clean
