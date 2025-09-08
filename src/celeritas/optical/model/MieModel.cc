#include "MieModel.hh"

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/InteractionApplier.hh"
#include "celeritas/optical/MfpBuilder.hh"
#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"
#include "celeritas/optical/model/MieExecutor.hh"

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
}  // [TD] might change the structure here

//---------------------------------------------------------------------------//
MieModel::MieModel(ActionId id, SPConstImported imported, Input input)
    : Model(id, "optical-mie", "interact by optical Mie scattering")
    , imported_(ImportModelClass::mie, std::move(imported))
    , input_(std::move(input))
{
    // CELER_EXPECT(!input_ || input_.materials->num_materials()
    //                        == imported_.num_materials());

    // for (auto mat : range(OptMatId(imported_.num_materials())))
    //{
    //     if (imported_.mfp(mat))
    //         CELER_LOG(debug)
    //             << "Mie: found imported MFP table for mat " << mat.get();
    //     else
    //         CELER_LOG(debug) << "Mie: no MFP table for mat " << mat.get()
    //                          << " (default = infinite MFP)";
    // }
}

//---------------------------------------------------------------------------//
void MieModel::build_mfps(OptMatId mat, MfpBuilder& build) const
{
    CELER_LOG(debug) << "MieModel::build_mfps called for mat " << mat.get();
    CELER_EXPECT(mat < imported_.num_materials());
    build(imported_.mfp(mat));
}

//---------------------------------------------------------------------------//
void MieModel::step(CoreParams const& params, CoreStateHost& state) const
{
    CELER_LOG(debug) << "MieModel::step called (not yet implemented)";
    launch_action(
        state,
        make_action_thread_executor(params.ptr<MemSpace::native>(),
                                    state.ptr(),
                                    this->action_id(),
                                    InteractionApplier{MieExecutor{}}));
}

#if !CELER_USE_DEVICE
void MieModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

}  // namespace optical
}  // namespace celeritas
