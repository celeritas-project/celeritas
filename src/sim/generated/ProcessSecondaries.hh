#include "base/Assert.hh"
#include "base/Macros.hh"
#include "sim/detail/ProcessSecondariesLauncher.hh"
#include "sim/TrackData.hh"
#include "sim/TrackInitData.hh"

namespace celeritas
{
namespace generated
{

void process_secondaries(
    const ParamsHostRef& params,
    const StateHostRef& states,
    const TrackInitStateHostRef& data);

void process_secondaries(
    const ParamsDeviceRef& params,
    const StateDeviceRef& states,
    const TrackInitStateDeviceRef& data);

#if !CELER_USE_DEVICE
inline void process_secondaries(const ParamsDeviceRef&, const StateDeviceRef&, const TrackInitStateDeviceRef&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

} // namespace generated
} // namespace celeritas
