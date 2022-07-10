#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/track/detail/LocateAliveLauncher.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/track/TrackInitData.hh"

namespace celeritas
{
namespace generated
{

void locate_alive(
    const CoreHostRef& core_data,
    const TrackInitStateHostRef& init_data);

void locate_alive(
    const CoreDeviceRef& core_data,
    const TrackInitStateDeviceRef& init_data);

#if !CELER_USE_DEVICE
inline void locate_alive(const CoreDeviceRef&, const TrackInitStateDeviceRef&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

} // namespace generated
} // namespace celeritas
