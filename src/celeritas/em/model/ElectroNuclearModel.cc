//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/ElectroNuclearModel.cc
//---------------------------------------------------------------------------//
#include "ElectroNuclearModel.hh"

#include "corecel/Types.hh"
#include "corecel/grid/VectorUtils.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/g4/EmExtraPhysicsHelper.hh"
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
ElectroNuclearModel::ElectroNuclearModel(ActionId id,
                                         ParticleParams const& particles,
                                         MaterialParams const& materials)
    : StaticConcreteAction(id, "electro-nuclear", "interact by electro-nuclear")
{
    CELER_EXPECT(id);

    HostVal<ElectroNuclearData> data;

    helper_ = std::make_shared<EmExtraPhysicsHelper>();

    // Save IDs
    data.scalars.electron_id = particles.find(pdg::electron());
    data.scalars.positron_id = particles.find(pdg::positron());

    CELER_VALIDATE(data.scalars.electron_id && data.scalars.positron_id,
                   << "missing particles (required for " << this->description()
                   << ")");

    // Electro-nuclear element cross section data
    NonuniformGridInserter insert_micro_xs{&data.reals, &data.micro_xs};

    double const emin = data.scalars.min_valid_energy().value();
    double const emax = data.scalars.max_valid_energy().value();
    for (auto el_id : range(ElementId{materials.num_elements()}))
    {
        AtomicNumber z = materials.get(el_id).atomic_number();

        // Build element cross sections
        insert_micro_xs(this->calc_micro_xs(z, emin, emax));
    }
    CELER_ASSERT(data.micro_xs.size() == materials.num_elements());

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<ElectroNuclearData>{std::move(data)};
    CELER_ENSURE(this->data_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto ElectroNuclearModel::applicability() const -> SetApplicability
{
    Applicability electron_nuclear_appli;
    electron_nuclear_appli.particle = this->host_ref().scalars.electron_id;
    electron_nuclear_appli.lower = this->host_ref().scalars.min_valid_energy();
    electron_nuclear_appli.upper = this->host_ref().scalars.max_valid_energy();

    Applicability positron_nuclear_appli = electron_nuclear_appli;
    positron_nuclear_appli.particle = this->host_ref().scalars.positron_id;

    return {electron_nuclear_appli, positron_nuclear_appli};
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto ElectroNuclearModel::micro_xs(Applicability) const -> XsTable
{
    // Cross sections are calculated on the fly
    return {};
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * Apply the interaction kernel.
 */
void ElectroNuclearModel::step(CoreParams const&, CoreStateHost&) const
{
    CELER_NOT_IMPLEMENTED("Electro-nuclear inelastic interaction");
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void ElectroNuclearModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
/*!
 * Build electro-nuclear element cross sections using G4ElectroNuclearXS.
 *
 * Tabulate cross sections using separate parameterizations for the high energy
 * region (emin < E < 50 GeV) and the ultra high energy region up to the
 * maximum valid energy (emax). The numbers of bins are chosen to adequately
 * capture both parameterized points (336 bins from 2.0612 MeV to 50 GeV) and
 * calculations used in G4ElectroNuclearCrossSection, which can be made
 * configurable if needed (TODO).
 */
inp::Grid ElectroNuclearModel::calc_micro_xs(AtomicNumber z,
                                             double emin,
                                             double emax) const
{
    CELER_EXPECT(z);

    inp::Grid result;

    // Upper limit of parameterizations of electro-nuclear cross section [Mev]
    double const emid = 5e+4;

    size_type nbin_total = 300;
    size_type nbin_ultra = 50;
    result.y.resize(nbin_total);

    result.x = geomspace(emin, emid, nbin_total - nbin_ultra);
    result.x.pop_back();
    auto ultra = geomspace(emid, emax, nbin_ultra + 1);
    result.x.insert(result.x.end(), ultra.begin(), ultra.end());

    // Tabulate the cross section from emin to emax
    for (size_type i = 0; i < nbin_total; ++i)
    {
        auto xs = helper_->calc_electro_nuclear_xs(
            z, MevEnergy{static_cast<real_type>(result.x[i])});
        result.y[i]
            = native_value_to<units::BarnXs>(native_value_from(xs)).value();
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
