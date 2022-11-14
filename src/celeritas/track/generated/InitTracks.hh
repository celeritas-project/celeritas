#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/track/detail/InitTracksLauncher.hh" // IWYU pragma: associated


namespace celeritas
{
namespace generated
{

void init_tracks(
    const CoreHostRef& core_data,
    const size_type num_vacancies);

void init_tracks(
    const CoreDeviceRef& core_data,
    const size_type num_vacancies);

#if !CELER_USE_DEVICE
inline void init_tracks(const CoreDeviceRef&, const size_type)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

} // namespace generated
} // namespace celeritas
