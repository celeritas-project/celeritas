//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/LivermorePEInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/StackAllocator.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/PolyEvaluator.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/LivermorePEData.hh"
#include "celeritas/em/xs/LivermorePEMicroXsCalculator.hh"
#include "celeritas/grid/NonuniformGridCalculator.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/InteractionUtils.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/Secondary.hh"

#include "AtomicRelaxationHelper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Livermore model for the photoelectric effect.
 *
 * A parameterization of the photoelectric cross sections in two different
 * energy intervals, formulated as
 * \f[
 * \sigma(E)
     = a_1 / E + a_2 / E^2 + a_3 / E^3 + a_4 / E^4 + a_5 / E^5 + a_6 / E^6
     \: ,
 * \f]
 * is used. The coefficients for this model are calculated by fitting the
 * tabulated EPICS2014 subshell cross sections. The parameterized model applies
 * above approximately 5 keV; below  this limit (which depends on the atomic
 * number) the tabulated cross sections are used. The angle of the emitted
 * photoelectron is sampled from the Sauter-Gavrila distribution.
 *
 * \note This performs the same sampling routine as in Geant4's
 * G4LivermorePhotoElectricModel class, as documented in section 6.3.5 of the
 * Geant4 Physics Reference (release 10.6).
 */
class LivermorePEInteractor
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    LivermorePEInteractor(NativeCRef<LivermorePEData> const& shared,
                          AtomicRelaxationHelper const& relaxation,
                          ElementId el_id,
                          ParticleTrackView const& particle,
                          CutoffView const& cutoffs,
                          Real3 const& inc_direction,
                          StackAllocator<Secondary>& allocate);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    //// DATA ////

    // Shared constant physics properties
    NativeCRef<LivermorePEData> const& shared_;
    // Shared scratch space
    AtomicRelaxationHelper const& relaxation_;
    // Index in MaterialParams elements
    ElementId el_id_;
    // Production thresholds
    CutoffView const& cutoffs_;
    // Incident direction
    Real3 const& inc_direction_;
    // Incident gamma energy
    real_type const inc_energy_;
    // Allocate space for one or more secondary particles
    StackAllocator<Secondary>& allocate_;
    // Microscopic cross section calculator
    LivermorePEMicroXsCalculator calc_micro_xs_;
    // Reciprocal of the energy
    real_type inv_energy_;

    //// HELPER FUNCTIONS ////

    // Sample the atomic subshel
    template<class Engine>
    inline CELER_FUNCTION SubshellId sample_subshell(Engine& rng) const;

    // Sample the direction of the emitted photoelectron
    template<class Engine>
    inline CELER_FUNCTION Real3 sample_direction(Engine& rng) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 *
 * The incident particle must be above the energy threshold: this should be
 * handled in code *before* the interactor is constructed.
 */
CELER_FUNCTION
LivermorePEInteractor::LivermorePEInteractor(
    NativeCRef<LivermorePEData> const& shared,
    AtomicRelaxationHelper const& relaxation,
    ElementId el_id,
    ParticleTrackView const& particle,
    CutoffView const& cutoffs,
    Real3 const& inc_direction,
    StackAllocator<Secondary>& allocate)
    : shared_(shared)
    , relaxation_(relaxation)
    , el_id_(el_id)
    , cutoffs_(cutoffs)
    , inc_direction_(inc_direction)
    , inc_energy_(value_as<Energy>(particle.energy()))
    , allocate_(allocate)
    , calc_micro_xs_(shared, particle.energy())
{
    CELER_EXPECT(particle.particle_id() == shared_.ids.gamma);
    CELER_EXPECT(inc_energy_ > 0);

    inv_energy_ = 1 / inc_energy_;
}

//---------------------------------------------------------------------------//
/*!
 * Sample using the Livermore model for the photoelectric effect.
 */
template<class Engine>
CELER_FUNCTION Interaction LivermorePEInteractor::operator()(Engine& rng)
{
    Span<Secondary> secondaries;
    size_type count = relaxation_ ? 1 + relaxation_.max_secondaries() : 1;
    if (Secondary* ptr = allocate_(count))
    {
        secondaries = {ptr, count};
    }
    else
    {
        // Failed to allocate space for secondaries
        return Interaction::from_failure();
    }

    // Sample atomic subshell
    SubshellId shell_id = this->sample_subshell(rng);

    // If the binding energy of the sampled shell is greater than the incident
    // photon energy, no secondaries are produced and the energy is deposited
    // locally.
    if (CELER_UNLIKELY(!shell_id))
    {
        Interaction result = Interaction::from_absorption();
        result.energy_deposition = Energy{inc_energy_};
        return result;
    }

    real_type binding_energy;
    {
        auto const& el = shared_.xs.elements[el_id_];
        auto const& shells = shared_.xs.shells[el.shells];
        binding_energy
            = value_as<Energy>(shells[shell_id.get()].binding_energy);
    }

    // Outgoing secondary is an electron
    CELER_ASSERT(!secondaries.empty());
    {
        Secondary& electron = secondaries.front();
        electron.particle_id = shared_.ids.electron;

        // Electron kinetic energy is the difference between the incident
        // photon energy and the binding energy of the shell
        electron.energy = Energy{inc_energy_ - binding_energy};

        // Direction of the emitted photoelectron is sampled from the
        // Sauter-Gavrila distribution
        electron.direction = this->sample_direction(rng);
    }

    // Construct interaction for change to primary (incident) particle
    Interaction result = Interaction::from_absorption();
    if (relaxation_)
    {
        // Sample secondaries from atomic relaxation, into all but the initial
        // secondary position
        AtomicRelaxation sample_relaxation = relaxation_.build_distribution(
            cutoffs_, shell_id, secondaries.subspan(1));

        auto outgoing = sample_relaxation(rng);
        secondaries = {secondaries.data(), 1 + outgoing.count};

        // The local energy deposition is the difference between the binding
        // energy of the vacancy subshell and the sum of the energies of any
        // secondaries created in atomic relaxation
        result.energy_deposition
            = Energy{binding_energy - value_as<Energy>(outgoing.energy)};
    }
    else
    {
        result.energy_deposition = Energy{binding_energy};
    }
    result.secondaries = secondaries;

    CELER_ENSURE(result.energy_deposition >= zero_quantity());
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Sample the shell from which the photoelectron is emitted.
 */
template<class Engine>
CELER_FUNCTION SubshellId LivermorePEInteractor::sample_subshell(Engine& rng) const
{
    LivermoreElement const& el = shared_.xs.elements[el_id_];
    auto const& shells = shared_.xs.shells[el.shells];
    size_type shell_id = 0;

    using Xs = RealQuantity<LivermoreSubshell::XsUnits>;
    real_type const cutoff = generate_canonical(rng)
                             * value_as<Xs>(calc_micro_xs_(el_id_));
    if (Energy{inc_energy_} < el.thresh_lo)
    {
        // Accumulate discrete PDF for tabulated shell cross sections
        //! \todo Rewrite using Selector with a remainder
        real_type xs = 0;
        real_type const inv_cube_energy = ipow<3>(inv_energy_);
        for (; shell_id < shells.size(); ++shell_id)
        {
            auto const& shell = shells[shell_id];
            if (Energy{inc_energy_} < shell.binding_energy)
            {
                // No chance of interaction because binding energy is higher
                // than incident
                continue;
            }

            // Use the tabulated subshell cross sections
            NonuniformGridCalculator calc_xs(shell.xs, shared_.xs.reals);
            xs += inv_cube_energy * calc_xs(inc_energy_);

            if (xs > cutoff)
            {
                break;
            }
        }

        if (CELER_UNLIKELY(shell_id == shells.size()))
        {
            // All shells are above incident energy (this can happen due to
            // a constant cross section below the lowest binding energy)
            return {};
        }
    }
    else
    {
        // Invert discrete CDF using a linear search. TODO: we could implement
        // an algorithm to encapsulate and later accelerate it.

        // Low/high index on params
        int const pidx = Energy{inc_energy_} < el.thresh_hi ? 0 : 1;
        size_type const shell_end = shells.size() - 1;

        for (; shell_id < shell_end; ++shell_id)
        {
            PolyEvaluator<real_type, 5> eval_poly(shells[shell_id].param[pidx]);

            // Calculate the *cumulative* subshell cross section (this plus all
            // below) from the fit parameters and energy as
            // \sigma(E) = a_1 / E + a_2 / E^2 + a_3 / E^3
            //             + a_4 / E^4 + a_5 / E^5 + a_6 / E^6.
            real_type xs = inv_energy_ * eval_poly(inv_energy_);
            if (xs > cutoff)
            {
                break;
            }
        }
    }

    return SubshellId{shell_id};
}

//---------------------------------------------------------------------------//
/*!
 * Sample a direction according to the Sauter-Gavrila distribution.
 *
 * \note The Sauter-Gavrila distribution for the K-shell is used to sample the
 * polar angle of a photoelectron. This performs the same sampling routine as
 * in Geant4's G4SauterGavrilaAngularDistribution class, as documented in
 * section 6.3.2 of the Geant4 Physics Reference (release 10.6) and section
 * 2.1.1.1 of the Penelope 2014 manual.
 */
template<class Engine>
CELER_FUNCTION Real3 LivermorePEInteractor::sample_direction(Engine& rng) const
{
    constexpr units::MevEnergy min_energy{1.e-6};
    constexpr units::MevEnergy max_energy{100.};

    if (inc_energy_ > value_as<Energy>(max_energy))
    {
        // If the incident gamma energy is above 100 MeV, use the incident
        // gamma direction for the direction of the emitted photoelectron.
        return inc_direction_;
    }
    // If the incident energy is below 1 eV, set it to 1 eV.
    real_type energy_per_mecsq = max(value_as<Energy>(min_energy), inc_energy_)
                                 * shared_.inv_electron_mass;

    // Calculate Lorentz factors of the photoelectron
    real_type gamma = energy_per_mecsq + 1;
    real_type beta = std::sqrt(energy_per_mecsq * (gamma + 1)) / gamma;
    real_type a = (1 - beta) / beta;

    // Second term inside the brackets in Eq. 2.8 in the Penelope manual
    constexpr real_type half = 0.5;
    real_type b = half * beta * gamma * energy_per_mecsq * (gamma - 2);

    // Maximum of the rejection function g(1 - cos \theta) given in Eq. 2.8,
    // which is attained when 1 - cos \theta = 0
    real_type g_max = 2 * (1 / a + b);

    // Rejection loop: sample 1 - cos \theta
    real_type g;
    real_type nu;
    do
    {
        // Sample 1 - cos \theta from the distribution given in Eq. 2.9 using
        // the inverse function (Eq. 2.11)
        real_type u = generate_canonical(rng);
        nu = 2 * a * (2 * u + (a + 2) * std::sqrt(u))
             / (ipow<2>(a + 2) - 4 * u);

        // Calculate the rejection function (Eq 2.8) at the sampled value
        g = (2 - nu) * (1 / (a + nu) + b);
    } while (g < g_max * generate_canonical(rng));

    // Sample the azimuthal angle and calculate the direction of the
    // photoelectron
    return ExitingDirectionSampler{1 - nu, inc_direction_}(rng);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
