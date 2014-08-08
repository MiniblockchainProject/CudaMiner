
if WANT_JANSSON
JANSSON_INCLUDES= -I$(top_srcdir)/compat/jansson
else
JANSSON_INCLUDES=
endif

EXTRA_DIST	= example-cfg.json nomacro.pl

SUBDIRS		= compat m7

INCLUDES	= $(PTHREAD_FLAGS) -fno-strict-aliasing $(JANSSON_INCLUDES)
INCLUDES   += -I$(top_srcdir)/m7

bin_PROGRAMS	= minerd

dist_man_MANS	= minerd.1

minerd_SOURCES	= elist.h miner.h compat.h \
		  cpu-miner.cpp util.cpp \
		  sha2.c scrypt.c m7hash.c \
		  trashminer.cu cuda_mul.cu \
		  cuda_sha256.cu cuda_sha512.cu \
		  cuda_whirlpool512.cu cuda_keccak512.cu \
		  cuda_tiger.cu cuda_ripemd160.cu \
		  cuda_haval256.cu \
		  cuda_check.cu

.cu.o:
	nvcc -Xptxas "-abi=no -v" -arch=compute_35 --ptxas-options=-v   -O2 -o $@ -c $<


if USE_ASM
if ARCH_x86
minerd_SOURCES += sha2-x86.S scrypt-x86.S
endif
if ARCH_x86_64
minerd_SOURCES += sha2-x64.S scrypt-x64.S
endif
if ARCH_ARM
minerd_SOURCES += sha2-arm.S scrypt-arm.S
endif
endif
minerd_LDFLAGS	= $(PTHREAD_FLAGS) -L/usr/local/cuda/lib64
minerd_LDADD	= @LIBCURL@ @JANSSON_LIBS@ @PTHREAD_LIBS@ @WS2_LIBS@ @M7_LIBS@ -lstdc++ -lcudart -lgmp
minerd_CPPFLAGS = @LIBCURL_CPPFLAGS@

