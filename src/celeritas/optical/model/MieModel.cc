#include "MieModel.hh"

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/Types.hh"
#include "celeritas/inp/Grid.hh"
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
    CELER_LOG(debug) << "MieModel::make_builder";

    for (size_type i = 0; i < input.data.size(); ++i)
    {
        ImportMie const& mie = input.data[i];

        CELER_LOG(debug) << "Material " << i << " (volume ID " << i << ")"
                         << " forward_g=" << mie.forward_g
                         << " backward_g=" << mie.backward_g
                         << " forward_ratio=" << mie.forward_ratio;
        //   << " scale_factor=" << mie.scale_factor;
        //<< " attenuation points=" << mie.attenuation.size();
    }

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
    , mie_data_(std::move(input.data))
{
    CELER_LOG(debug) << "Miemodel constructor";
    CELER_LOG(debug) << "MieModel registered with action ID "
                     << this->action_id().get();

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
    if (auto const& mfp = imported_.mfp(mat))
    {
        CELER_LOG(debug) << "mie model mfp found" << mat.get()
                         << " with the following mfps " << &mfp;
        build(mfp);
    }
    else
    {
        CELER_LOG(debug) << "mie model MFP not found for " << mat.get();
        inp::Grid g;
        g.x = {1.56962, 6.19998};
        g.y = {std::numeric_limits<real_type>::infinity(),
               std::numeric_limits<real_type>::infinity()};
        build(g);
    }
}
//    build(imported_.mfp(mat));

//---------------------------------------------------------------------------//
void MieModel::step(CoreParams const& params, CoreStateHost& state) const
{
    CELER_LOG(debug) << "MieModel::step called ";
    launch_action(state,
                  make_action_thread_executor(
                      params.ptr<MemSpace::native>(),
                      state.ptr(),
                      this->action_id(),
                      InteractionApplier{MieExecutor{mie_data_}}));
}

#if !CELER_USE_DEVICE
void MieModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

}  // namespace optical
}  // namespace celeritas
