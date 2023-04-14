#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/global/CoreTrackData.hh"


namespace celeritas
{
namespace generated
{

void locate_alive(
    HostCRef<CoreParamsData> const& core_params,
    HostRef<CoreStateData> const& core_states);

void locate_alive(
    DeviceCRef<CoreParamsData> const& core_params,
    DeviceRef<CoreStateData> const& core_states);

#if !CELER_USE_DEVICE
inline void locate_alive(DeviceCRef<CoreParamsData> const&, DeviceRef<CoreStateData> const&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

}  // namespace generated
}  // namespace celeritas
