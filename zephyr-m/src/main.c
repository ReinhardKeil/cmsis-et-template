/*
 * Copyright (c) 2012-2014 Wind River Systems, Inc.
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>

#include "executorch_test.h"

int main(void)
{
	printf("Hello World! %s\n", CONFIG_BOARD_TARGET);
	return executorch_run_test();
}
