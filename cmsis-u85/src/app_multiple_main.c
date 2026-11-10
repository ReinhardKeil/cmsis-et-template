/* Copyright 2026 Arm Limited and/or its affiliates.
 * SPDX-License-Identifier: Apache-2.0 */
#include <stdio.h>

int executorch_run_test(void);

int app_main(void)
{
  const int result = executorch_run_test();
  putchar('\x04'); /* EOT: the supplied CMSIS examples run on an FVP. */
  fflush(stdout);
  return result;
}
