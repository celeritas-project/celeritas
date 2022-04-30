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
