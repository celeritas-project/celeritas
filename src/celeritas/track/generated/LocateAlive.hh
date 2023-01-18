#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/global/CoreTrackData.hh"


namespace celeritas
{
namespace generated
{

void locate_alive(
    CoreHostRef const& core_data);

void locate_alive(
    CoreDeviceRef const& core_data);

#if !CELER_USE_DEVICE
inline void locate_alive(CoreDeviceRef const&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

}  // namespace generated
}  // namespace celeritas
