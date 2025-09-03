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
    // CELER_EXPECT(!input_ || input_.materials->num_materials()
    //                        == imported_.num_materials());

    for (auto mat : range(OptMatId(imported_.num_materials())))
    {
        if (imported_.mfp(mat))
            CELER_LOG(debug)
                << "Mie: found imported MFP table for mat " << mat.get();
        else
            CELER_LOG(debug) << "Mie: no MFP table for mat " << mat.get()
                             << " (default = infinite MFP)";
    }
}

//---------------------------------------------------------------------------//
void MieModel::build_mfps(OptMatId mat, MfpBuilder& build) const
{
    // CELER_EXPECT(mat < imported_.num_materials());
    CELER_LOG(debug) << "MieModel::build_mfps called for mat " << mat.get();
    if (auto const& mfp = imported_.mfp(mat))
    {
        build(mfp);
    }
    // else
    // {
    //     build();  // empty grid → infinite mean free path
    // }
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
