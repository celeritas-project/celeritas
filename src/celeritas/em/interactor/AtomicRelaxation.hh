//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/AtomicRelaxation.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/MiniStack.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/AtomicRelaxationData.hh"
#include "celeritas/phys/CutoffView.hh"
#include "celeritas/random/distribution/IsotropicDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Atomic relaxation.
 *
 * The EADL radiative and non-radiative transition data is used to simulate the
 * emission of fluorescence photons and (optionally) Auger electrons given an
 * initial shell vacancy created by a primary process.
 */
class AtomicRelaxation
{
  public:
    //!@{
    //! Type aliases
    using MevEnergy = units::MevEnergy;
    //!@}

    struct result_type
    {
        size_type count{};  //!< Number of secondaries created
        real_type energy{}; //!< Sum of the energies of the secondaries
    };

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION AtomicRelaxation(const AtomicRelaxParamsRef& shared,
                                           const CutoffView&           cutoffs,
                                           ElementId                   el_id,
                                           SubshellId       shell_id,
                                           Span<Secondary>  secondaries,
                                           Span<SubshellId> vacancies);

    // Simulate atomic relaxation with an initial vacancy in the given shell ID
    template<class Engine>
    inline CELER_FUNCTION result_type operator()(Engine& rng);

  private:
    // Shared EADL atomic relaxation data
    const AtomicRelaxParamsRef& shared_;
    // Photon production threshold [MeV]
    real_type gamma_cutoff_;
    // Electron production threshold [MeV]
    real_type electron_cutoff_;
    // Index in MaterialParams elements
    ElementId el_id_;
    // Shell ID of the initial vacancy
    SubshellId shell_id_;
    // Fluorescence photons and Auger electrons
    Span<Secondary> secondaries_;
    // Storage for stack of unprocessed subshell vacancies
    Span<SubshellId> vacancies_;
    // Angular distribution of secondaries
    IsotropicDistribution<real_type> sample_direction_;

  private:
    using TransitionId = OpaqueId<AtomicRelaxTransition>;

    template<class Engine>
    inline CELER_FUNCTION TransitionId
    sample_transition(const AtomicRelaxSubshell& shell, Engine& rng);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 *
 * The secondaries must have enough storage allocated for particles produced in
 * atomic relaxation and the vacancies must have enough storage allocated for
 * the stack of subshell IDs: this should be handled in code *before*
 * construction.
 *
 * The precondition of the element having relaxation data is satisfied by the
 * AtomicRelaxationHelper -- it is only "true" if a distribution can be
 * emitted.
 */
CELER_FUNCTION
AtomicRelaxation::AtomicRelaxation(const AtomicRelaxParamsRef& shared,
                                   const CutoffView&           cutoffs,
                                   ElementId                   el_id,
                                   SubshellId                  shell_id,
                                   Span<Secondary>             secondaries,
                                   Span<SubshellId>            vacancies)
    : shared_(shared)
    , gamma_cutoff_(cutoffs.energy(shared_.ids.gamma).value())
    , electron_cutoff_(cutoffs.energy(shared_.ids.electron).value())
    , el_id_(el_id)
    , shell_id_(shell_id)
    , secondaries_(secondaries)
    , vacancies_(vacancies)
{
    CELER_EXPECT(shared_ && el_id_ < shared_.elements.size()
                 && shared_.elements[el_id_]);
    CELER_EXPECT(shell_id);
}

//---------------------------------------------------------------------------//
/*!
 * Simulate atomic relaxation with an initial vacancy in the given shell ID
 */
template<class Engine>
CELER_FUNCTION AtomicRelaxation::result_type
AtomicRelaxation::operator()(Engine& rng)
{
    const AtomicRelaxElement& el     = shared_.elements[el_id_];
    const auto&               shells = shared_.shells[el.shells];
    MiniStack<SubshellId>     vacancies(vacancies_);

    // Push the vacancy created by the primary process onto a stack.
    vacancies.push(shell_id_);

    // Total number of secondaries
    size_type count      = 0;
    real_type sum_energy = 0;

    // Generate the shower of photons and electrons produced by radiative and
    // non-radiative transitions
    while (!vacancies.empty())
    {
        // Pop the vacancy off the stack and check if it has transition data
        SubshellId vacancy_id = vacancies.pop();
        if (vacancy_id.get() >= shells.size())
            continue;

        // Sample a transition (TODO: refactor to use Selector but with
        // "remainder")
        const AtomicRelaxSubshell& shell = shells[vacancy_id.get()];
        const TransitionId trans_id      = this->sample_transition(shell, rng);

        if (!trans_id)
            continue;

        // Push the new vacancies onto the stack and create the secondary
        const auto& transition
            = shared_.transitions[shell.transitions][trans_id.get()];
        vacancies.push(transition.initial_shell);
        if (transition.auger_shell)
        {
            vacancies.push(transition.auger_shell);

            if (transition.energy >= electron_cutoff_)
            {
                // Sampled a non-radiative transition: create an Auger electron
                CELER_ASSERT(count < secondaries_.size());
                Secondary& secondary  = secondaries_[count++];
                secondary.direction   = sample_direction_(rng);
                secondary.energy      = MevEnergy{transition.energy};
                secondary.particle_id = shared_.ids.electron;

                // Accumulate the energy carried away by secondaries
                sum_energy += transition.energy;
            }
        }
        else if (transition.energy >= gamma_cutoff_)
        {
            // Sampled a radiative transition: create a fluorescence photon
            CELER_ASSERT(count < secondaries_.size());
            Secondary& secondary  = secondaries_[count++];
            secondary.direction   = sample_direction_(rng);
            secondary.energy      = MevEnergy{transition.energy};
            secondary.particle_id = shared_.ids.gamma;

            // Accumulate the energy carried away by secondaries
            sum_energy += transition.energy;
        }
    }

    result_type result;
    result.count  = count;
    result.energy = sum_energy;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Sample an atomic transition.
 *
 * TODO: refactor to use a Selector-like algorithm that allows a "remainder"
 * that indicates "not sampled".
 */
template<class Engine>
inline CELER_FUNCTION auto
AtomicRelaxation::sample_transition(const AtomicRelaxSubshell& shell,
                                    Engine& rng) -> TransitionId
{
    const auto& transitions = shared_.transitions[shell.transitions];

    real_type accum = -generate_canonical(rng);
    for (size_type i = 0; i < transitions.size(); ++i)
    {
        accum += transitions[i].probability;
        if (accum > 0)
            return TransitionId{i};
    }

    // No transition was sampled: skip to the next vacancy
    return {};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
