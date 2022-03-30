#include "base/Assert.hh"
#include "base/Macros.hh"
#include "sim/detail/LocateAliveLauncher.hh"
#include "sim/CoreTrackData.hh"
#include "sim/TrackInitData.hh"

namespace celeritas
{
namespace generated
{

void locate_alive(
    const CoreParamsHostRef& params,
    const CoreStateHostRef& states,
    const TrackInitStateHostRef& data);

void locate_alive(
    const CoreParamsDeviceRef& params,
    const CoreStateDeviceRef& states,
    const TrackInitStateDeviceRef& data);

#if !CELER_USE_DEVICE
inline void locate_alive(const CoreParamsDeviceRef&, const CoreStateDeviceRef&, const TrackInitStateDeviceRef&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

} // namespace generated
} // namespace celeritas
