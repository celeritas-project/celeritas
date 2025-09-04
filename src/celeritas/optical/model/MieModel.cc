#include "MieModel.hh"

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/optical/MfpBuilder.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
auto MieModel::make_builder(SPConstImported imported, Input input)
    -> ModelBuilder
{
    CELER_EXPECT(imported);
    return [imported = std::move(imported),
            input = std::move(input)](ActionId id) {
        return std::make_shared<MieModel>(id, imported, input);
    };
}

//---------------------------------------------------------------------------//
MieModel::MieModel(ActionId id, SPConstImported imported, Input input)
    : Model(id, "optical-mie", "interact by optical Mie scattering")
    , imported_(ImportModelClass::mie, std::move(imported))
    , input_(std::move(input))
{
}

//---------------------------------------------------------------------------//
void MieModel::build_mfps(OptMatId mat, MfpBuilder& build) const
{
    CELER_EXPECT(mat < imported_.num_materials());
    build(imported_.mfp(mat));
}

//---------------------------------------------------------------------------//
void MieModel::step(CoreParams const&, CoreStateHost&) const
{
    CELER_LOG(debug) << "MieModel::step called (not yet implemented)";
}

#if !CELER_USE_DEVICE
void MieModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

}  // namespace optical
}  // namespace celeritas
