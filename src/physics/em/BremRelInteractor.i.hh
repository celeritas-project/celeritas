//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BremRelInteractor.i.hh
//---------------------------------------------------------------------------//

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
BremRelInteractor::BremRelInteractor(const BremRelInteractorPointers& shared,
                                     const ParticleTrackView&         particle,
                                     const MaterialTrackView&         mat,
                                     const Real3&            inc_direction,
                                     SecondaryAllocatorView& allocate)
    : shared_(shared)
    , allocate_(allocate)
    , mat_(mat)
    , inc_energy_(particle.energy())
    , inc_direction_(inc_direction)
    , use_lpm_(shared.use_lpm)
{
    REQUIRE(inc_energy_ >= this->min_incident_energy()
            && inc_energy_ <= this->max_incident_energy());
    REQUIRE(particle.def_id() == shared_.gamma_id); // XXX

    real_type density_factor = shared_.migdal_constant
                               * mat_.electron_density();
    if (use_lpm_)
    {
        real_type threshold = std::sqrt(density_factor) * this->lpm_energy();
        if (particle.energy().value() < threshold)
        {
            // Energy is below material-based cutoff
            use_lpm_ = false;
        }
    }

    // Total energy: rest mass * c^2 + kinetic energy
    real_type total_e = particle.energy().value() + particle.mass().value();
    density_corr_     = density_factor * total_e * total_e;

    ENSURE(density_corr_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Sample using the XXX model.
 */
template<class Engine>
CELER_FUNCTION Interaction BremRelInteractor::operator()(Engine& rng)
{
#if 0
    // Select target atom
    MatComponentSelector select_component(mat_, xs_);
    MatElementId         component_id = select_element(rng);
    ElementId            el           = mat_.element(component_id);
#endif

    // Allocate space for XXX (electron, multiple particles, ...)
    Secondary* secondaries = this->allocate_(0); // XXX
    if (secondaries == nullptr)
    {
        // Failed to allocate space for a secondary
        return Interaction::from_failure();
    }

    // XXX sample
    (void)sizeof(rng);

    // Construct interaction for change to primary (incident) particle
    Interaction result;
    result.action      = Action::scattered;                     // XXX
    result.energy      = units::MevEnergy{inc_energy_.value()}; // XXX
    result.direction   = inc_direction_;
    result.secondaries = {secondaries, 1}; // XXX

    // Save outgoing secondary data
    secondaries[0].def_id    = shared_.electron_id; // XXX
    secondaries[0].energy    = units::MevEnergy{0}; // XXX
    secondaries[0].direction = {0, 0, 0};           // XXX

    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
