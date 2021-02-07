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
 * atomic relaxation: this should be handled in code *before* construction.
 */
CELER_FUNCTION
AtomicRelaxation::AtomicRelaxation(const AtomicRelaxParamsPointers& shared,
                                   ElementId                        el_id,
                                   SubshellId                       shell_id,
                                   Span<Secondary> secondaries,
                                   size_type       base_size)
    : shared_(shared)
    , el_id_(el_id)
    , shell_id_(shell_id)
    , secondaries_(secondaries)
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
    // TODO: How should we allocate storage for these vacancies? Possibly use
    // StackAllocator? If we are only simulating radiative transitions, there
    // is only ever 1 vacancy waiting to be processed. For non-radiative
    // transitions, the upper bound on the maximum number of vacancies in the
    // stack at one time is n, where n is the number of shells containing
    // transition data for a given element (19 for Z = 100). But in practice
    // that won't happen, and we could probably bound it closer to 5 for even
    // the highest Z.
    Array<SubshellId, 10> vacancy_storage;
    MiniStack<SubshellId> vacancies(make_span(vacancy_storage));
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
        {
            continue;
        }

        // Sample a transition
        size_type                  i;
        real_type                  prob = generate_canonical(rng);
        const AtomicRelaxSubshell& shell
            = shared_.elements[el_id_.get()].shells[vacancy_id.get()];
        for (i = 0; i < shell.transitions.size(); ++i)
        {
            if ((prob -= shell.transitions[i].probability) <= 0)
            {
                break;
            }
        }

        // If no transition was sampled, continue to the next vacancy
        if (prob > 0.)
        {
            continue;
        }
        const AtomicRelaxTransition& transition = shell.transitions[i];

        CELER_ASSERT(count < secondaries_.size());
        Secondary& secondary = secondaries_[count++];
        secondary.direction  = sample_direction_(rng);
        secondary.energy     = MevEnergy{transition.energy};
        vacancies.push(transition.initial_shell);
        if (!transition.auger_shell)
        {
            // Sampled a radiative transition: create a fluorescence photon
            secondary.particle_id = shared_.gamma_id;
        }
        else
        {
            // Sampled a non-radiative transition: create an Auger electron
            secondary.particle_id = shared_.electron_id;
            vacancies.push(transition.auger_shell);
        }

        // Accumulate the energy carried away by secondaries
        result.energy += transition.energy;
    }

    result.secondaries = {secondaries_.data(), count};
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
