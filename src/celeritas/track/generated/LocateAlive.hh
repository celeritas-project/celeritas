#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/global/CoreTrackData.hh"


namespace celeritas
{
namespace generated
{

void locate_alive(
    const CoreHostRef& core_data);

void locate_alive(
    const CoreDeviceRef& core_data);

#if !CELER_USE_DEVICE
inline void locate_alive(const CoreDeviceRef&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

} // namespace generated
} // namespace celeritas
