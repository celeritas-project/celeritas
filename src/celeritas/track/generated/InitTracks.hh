#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/track/detail/InitTracksLauncher.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/track/TrackInitData.hh"

namespace celeritas
{
namespace generated
{

void init_tracks(
    const CoreHostRef& core_data,
    const TrackInitStateHostRef& init_data,
    const size_type num_vacancies);

void init_tracks(
    const CoreDeviceRef& core_data,
    const TrackInitStateDeviceRef& init_data,
    const size_type num_vacancies);

#if !CELER_USE_DEVICE
inline void init_tracks(const CoreDeviceRef&, const TrackInitStateDeviceRef&, const size_type)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

} // namespace generated
} // namespace celeritas
