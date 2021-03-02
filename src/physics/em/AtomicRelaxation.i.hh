//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file AtomicRelaxation.i.hh
//---------------------------------------------------------------------------//

#include "base/MiniStack.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 *
 * The secondaries must have enough storage allocated for particles produced in
 * atomic relaxation and the vacancies must have enough storage allocated for
 * the stack of subshell IDs: this should be handled in code *before*
 * construction.
 */
CELER_FUNCTION
AtomicRelaxation::AtomicRelaxation(const AtomicRelaxParamsPointers& shared,
                                   ElementId                        el_id,
                                   SubshellId                       shell_id,
                                   Span<Secondary>  secondaries,
                                   Span<SubshellId> vacancies,
                                   size_type        base_size)
    : shared_(shared)
    , el_id_(el_id)
    , shell_id_(shell_id)
    , secondaries_(secondaries)
    , vacancies_(vacancies)
    , base_size_(base_size)
{
    CELER_EXPECT(!shared_ || el_id_ < shared_.elements.size());
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
    // Particles produced and energy removed by secondaries
    result_type result{{secondaries_.data(), base_size_}, 0.};

    // Atomic relaxation is off *or* there is no transition data for this
    // element (true for Z < 6) *or* there is no transition data for this shell
    if (!shared_ || !shared_.elements[el_id_.get()]
        || shell_id_.get() >= shared_.elements[el_id_.get()].shells.size())
    {
        return result;
    }

    // Push the vacancy created by the primary process onto a stack
    MiniStack<SubshellId> vacancies(vacancies_);
    vacancies.push(shell_id_);

    // Total number of secondaries
    size_type count = base_size_;

    // Generate the shower of photons and electrons produced by radiative and
    // non-radiative transitions
    while (!vacancies.empty())
    {
        // Pop the vacancy off the stack and check if it has transition data
        SubshellId vacancy_id = vacancies.pop();
        if (!vacancy_id)
            continue;

        // Sample a transition
        const AtomicRelaxSubshell& shell
            = shared_.elements[el_id_.get()].shells[vacancy_id.get()];
        real_type prob = generate_canonical(rng);
        size_type i;
        for (i = 0; i < shell.transitions.size(); ++i)
        {
            if ((prob -= shell.transitions[i].probability) <= 0)
                break;
        }

        // If no transition was sampled, continue to the next vacancy
        if (prob > 0.)
            continue;

        // Push the new vacancies onto the stack and create the secondary
        const AtomicRelaxTransition& transition = shell.transitions[i];
        vacancies.push(transition.initial_shell);
        if (transition.auger_shell)
        {
            vacancies.push(transition.auger_shell);

            if (transition.energy >= shared_.electron_cut.value())
            {
                // Sampled a non-radiative transition: create an Auger electron
                CELER_ASSERT(count < secondaries_.size());
                Secondary& secondary  = secondaries_[count++];
                secondary.direction   = sample_direction_(rng);
                secondary.energy      = MevEnergy{transition.energy};
                secondary.particle_id = shared_.electron_id;
            }
        }
        else if (transition.energy >= shared_.gamma_cut.value())
        {
            // Sampled a radiative transition: create a fluorescence photon
            CELER_ASSERT(count < secondaries_.size());
            Secondary& secondary  = secondaries_[count++];
            secondary.direction   = sample_direction_(rng);
            secondary.energy      = MevEnergy{transition.energy};
            secondary.particle_id = shared_.gamma_id;
        }

        // Accumulate the energy carried away by secondaries
        result.energy += transition.energy;
    }

    result.secondaries = {secondaries_.data(), count};
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
