#include "GaussianRoughnessModel.hh"
#include "PolishedRoughnessModel.hh"
#include "SmearRoughnessModel.hh"

namespace celeritas
{
namespace optical
{

GaussianRoughnessModelController::GaussianRoughnessModelController(
    std::vector<inp::GaussianRoughness> const& input)
{
    HostVal<GaussianRoughnessData> data;
    auto build_sigma_alpha = ::celeritas::make_builder(&data.sigma_alpha);

    for (auto const& gaussian : input)
    {
        CELER_ENSURE(gaussian);
        build_sigma_alpha.push_back(gaussian.sigma_alpha);
    }

    CELER_ENSURE(data);
    CELER_ENSURE(data.sigma_alpha.size() == input.size());

    data_ = CollectionMirror<GaussianRoughnessData>{std::move(data)};
}

SmearRoughnessModelController::SmearRoughnessModelController(
    std::vector<inp::SmearRoughness> const& input)
{
    HostVal<SmearRoughnessData> data;
    auto build_roughness = ::celeritas::make_builder(&data.roughness);

    for (auto const& smear : input)
    {
        CELER_ENSURE(smear);
        build_roughness.push_back(smear.roughness);
    }

    CELER_ENSURE(data);
    CELER_ENSURE(data.roughness.size() == input.size());

    data_ = CollectionMirror<SmearRoughnessData>{std::move(data)};
}

}  // namespace optical
}  // namespace celeritas
