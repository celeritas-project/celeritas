#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/global/CoreTrackData.hh"


namespace celeritas
{
namespace generated
{

void init_tracks(
    HostCRef<CoreParamsData> const& core_params,
    HostRef<CoreStateData> const& core_states,
    size_type const num_vacancies);

void init_tracks(
    DeviceCRef<CoreParamsData> const& core_params,
    DeviceRef<CoreStateData> const& core_states,
    size_type const num_vacancies);

#if !CELER_USE_DEVICE
inline void init_tracks(DeviceCRef<CoreParamsData> const&, DeviceRef<CoreStateData> const&, size_type const)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

}  // namespace generated
}  // namespace celeritas
