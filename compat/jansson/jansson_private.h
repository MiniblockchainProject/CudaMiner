/*
 * Copyright (c) 2009, 2010 Petri Lehtinen <petri@digip.org>
 *
 * Jansson is free software; you can redistribute it and/or modify
 * it under the terms of the MIT license. See LICENSE for details.
 */

#ifndef JANSSON_PRIVATE_H
#define JANSSON_PRIVATE_H

#include "jansson.h"
#include "hashtable.h"

#include <stdio.h>
#include <stdarg.h>

#ifdef _MSC_VER

#define snprintf c99_snprintf

static int c99_snprintf(char* str, size_t size, const char* format, ...)
{
    int count;
    va_list ap;

    va_start(ap, format);
    count = c99_vsnprintf(str, size, format, ap);
    va_end(ap);

    return count;
}

static int c99_vsnprintf(char* str, size_t size, const char* format, va_list ap)
{
    int count = -1;

    if (size != 0)
        count = _vsnprintf_s(str, size, _TRUNCATE, format, ap);
    if (count == -1)
        count = _vscprintf(format, ap);

    return count;
}

#endif // _MSC_VER

#define container_of(ptr_, type_, member_)  \
    ((type_ *)((char *)ptr_ - (size_t)&((type_ *)0)->member_))

typedef struct {
    json_t json;
    hashtable_t hashtable;
    unsigned long serial;
    int visited;
} json_object_t;

typedef struct {
    json_t json;
    unsigned int size;
    unsigned int entries;
    json_t **table;
    int visited;
} json_array_t;

typedef struct {
    json_t json;
    char *value;
} json_string_t;

typedef struct {
    json_t json;
    double value;
} json_real_t;

typedef struct {
    json_t json;
    int value;
} json_integer_t;

#define json_to_object(json_)  container_of(json_, json_object_t, json)
#define json_to_array(json_)   container_of(json_, json_array_t, json)
#define json_to_string(json_)  container_of(json_, json_string_t, json)
#define json_to_real(json_)   container_of(json_, json_real_t, json)
#define json_to_integer(json_) container_of(json_, json_integer_t, json)

typedef struct {
    unsigned long serial;
    char key[];
} object_key_t;

const object_key_t *jsonp_object_iter_fullkey(void *iter);

#endif
