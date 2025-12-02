//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/GammaNuclearModel.cc
//---------------------------------------------------------------------------//
#include "GammaNuclearModel.hh"

#include "corecel/math/Quantity.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/grid/NonuniformGridInserter.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/InteractionApplier.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
GammaNuclearModel::GammaNuclearModel(ActionId id,
                                     ParticleParams const& particles,
                                     MaterialParams const& materials,
                                     ReadData load_data)
    : StaticConcreteAction(id, "gamma-nuclear", "interact by gamma-nuclear")
{
    CELER_EXPECT(id);
    CELER_EXPECT(load_data);

    HostVal<GammaNuclearData> data;

    // Save IDs
    data.scalars.gamma_id = particles.find(pdg::gamma());

    CELER_VALIDATE(data.scalars.gamma_id,
                   << "missing gamma (required for " << this->description()
                   << ")");

    // Save particle properties
    CELER_EXPECT(data.scalars);

    // Load gamma-nuclear element cross section data
    NonuniformGridInserter insert_micro_xs{&data.reals, &data.micro_xs};
    for (auto el_id : range(ElementId{materials.num_elements()}))
    {
        AtomicNumber z = materials.get(el_id).atomic_number();
        insert_micro_xs(load_data(z));
    }
    CELER_ASSERT(data.micro_xs.size() == materials.num_elements());

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<GammaNuclearData>{std::move(data)};
    CELER_ENSURE(this->data_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto GammaNuclearModel::applicability() const -> SetApplicability
{
    Applicability gamma_nuclear_applic;
    gamma_nuclear_applic.particle = this->host_ref().scalars.gamma_id;
    gamma_nuclear_applic.lower = zero_quantity();
    gamma_nuclear_applic.upper = this->host_ref().scalars.max_valid_energy();

    return {gamma_nuclear_applic};
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto GammaNuclearModel::micro_xs(Applicability) const -> XsTable
{
    // Cross sections are calculated on the fly
    return {};
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * Apply the interaction kernel.
 */
void GammaNuclearModel::step(CoreParams const&, CoreStateHost&) const
{
    CELER_NOT_IMPLEMENTED("Gamma-nuclear inelastic interaction");
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void GammaNuclearModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
