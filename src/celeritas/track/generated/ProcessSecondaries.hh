#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/track/detail/ProcessSecondariesLauncher.hh" // IWYU pragma: associated
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/track/TrackInitData.hh"

namespace celeritas
{
namespace generated
{

void process_secondaries(
    const CoreHostRef& core_data,
    const TrackInitStateHostRef& init_data);

void process_secondaries(
    const CoreDeviceRef& core_data,
    const TrackInitStateDeviceRef& init_data);

#if !CELER_USE_DEVICE
inline void process_secondaries(const CoreDeviceRef&, const TrackInitStateDeviceRef&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

} // namespace generated
} // namespace celeritas
