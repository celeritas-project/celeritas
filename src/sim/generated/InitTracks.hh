#include "base/Assert.hh"
#include "base/Macros.hh"
#include "sim/detail/InitTracksLauncher.hh"
#include "sim/CoreTrackData.hh"
#include "sim/TrackInitData.hh"

namespace celeritas
{
namespace generated
{

void init_tracks(
    const CoreParamsHostRef& params,
    const CoreStateHostRef& states,
    const TrackInitStateHostRef& data,
    const size_type num_vacancies);

void init_tracks(
    const CoreParamsDeviceRef& params,
    const CoreStateDeviceRef& states,
    const TrackInitStateDeviceRef& data,
    const size_type num_vacancies);

#if !CELER_USE_DEVICE
inline void init_tracks(const CoreParamsDeviceRef&, const CoreStateDeviceRef&, const TrackInitStateDeviceRef&, const size_type)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

} // namespace generated
} // namespace celeritas
