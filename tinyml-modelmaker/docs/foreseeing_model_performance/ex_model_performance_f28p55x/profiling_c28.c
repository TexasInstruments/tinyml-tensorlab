#include "cputimer.h"
#include "profiling.h"

#define CPUTIMER_MAX_PERIOD (0xFFFFFFFFUL)

uint32_t TEUtils_cycle_init(void)
{
    CPUTimer_clearOverflowFlag(CPUTIMER1_BASE);

    CPUTimer_setPeriod(CPUTIMER1_BASE, CPUTIMER_MAX_PERIOD);

    CPUTimer_setPreScaler(CPUTIMER1_BASE, 0UL);

    CPUTimer_reloadTimerCounter(CPUTIMER1_BASE);

    CPUTimer_stopTimer(CPUTIMER1_BASE);

    CPUTimer_startTimer(CPUTIMER1_BASE);

    return 0;
}

void TEUtils_cycle_stop(void)
{
    CPUTimer_stopTimer(CPUTIMER1_BASE);
}

// With higher levels of optimization, it’s possible that the compiler moves
// application code before the first call to CycleCounter_Read() or after the
// second call to CycleCounter_Read(). This can result in the reported cycle
// count being lower than the actual cycle count.
// Disabling inlining of CycleCounter_Read prevents this from occurring.
#pragma FUNC_CANNOT_INLINE(TEUtils_cycle_read)
uint32_t TEUtils_cycle_read(void)
{
    return CPUTimer_getTimerCount(CPUTIMER1_BASE);
}


uint32_t TEUtils_cycle_diff(uint32_t t0, uint32_t t1)
{
    return t0 - t1;
}
