#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/global/CoreTrackData.hh"


namespace celeritas
{
namespace generated
{

void process_secondaries(
    const CoreHostRef& core_data);

void process_secondaries(
    const CoreDeviceRef& core_data);

#if !CELER_USE_DEVICE
inline void process_secondaries(const CoreDeviceRef&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

} // namespace generated
} // namespace celeritas
