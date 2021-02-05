//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file AtomicRelaxation.i.hh
//---------------------------------------------------------------------------//

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
                                   Span<Secondary> secondaries)
    : shared_(shared)
    , el_id_(el_id)
    , secondaries_(secondaries)
    , num_vacancies_(0)
{
    CELER_EXPECT(el_id_ < shared_.elements.size());
    CELER_EXPECT(secondaries_.size()
                 >= shared_.elements[el_id_.get()].max_secondary);
}

//---------------------------------------------------------------------------//
/*!
 * Simulate atomic relaxation with an initial vacancy in the given shell ID and
 * return the total number of secondaries created.
 */
template<class Engine>
CELER_FUNCTION size_type AtomicRelaxation::operator()(SubshellId shell_id,
                                                      Engine&    rng)
{
    // The EADL only provides transition probabilities for 6 <= Z <= 100, so
    // atomic relaxation is not applicable for Z < 6. Also, transitions are
    // only provided for K, L, M, N, and some O shells.
    const AtomicRelaxElement& el = shared_.elements[el_id_.get()];
    if (!el || shell_id.get() >= el.shells.size())
    {
        return 0;
    }

    // Push the vacancy created by the primary process onto the stack
    vacancies_[num_vacancies_++] = shell_id;

    // Total number of particles produced
    size_type count = 0;

    // Generate the shower of photons and electrons produced by radiative and
    // non-radiative transitions
    while (num_vacancies_)
    {
        // Pop the vacancy off the stack
        SubshellId vacancy_id = vacancies_[--num_vacancies_];

        // If there are no transitions, process the next vacancy
        if (!vacancy_id)
        {
            continue;
        }

        // Sample the transition
        size_type                  i;
        real_type                  prob  = generate_canonical(rng);
        const AtomicRelaxSubshell& shell = el.shells[vacancy_id.get()];
        for (i = 0; i < shell.transitions.size(); ++i)
        {
            if ((prob -= shell.transitions[i].probability) <= 0)
            {
                break;
            }
        }

        // If no transition was sampled, continue to the next vacancy;
        // otherwise get the sampled transition
        if (prob > 0.)
        {
            continue;
        }
        const AtomicRelaxTransition& transition = shell.transitions[i];

        if (transition.auger_shell)
        {
            // If no Auger subshell is provided, this is a radiative
            // transition: create a fluorescence photon
            CELER_ASSERT(count < secondaries_.size());
            secondaries_[count].particle_id = shared_.gamma_id;
            secondaries_[count].direction   = sample_direction_(rng);
            secondaries_[count++].energy    = MevEnergy{transition.energy};

            // Push the new vacancy onto the stack
            CELER_ASSERT(num_vacancies_ < vacancies_.size());
            vacancies_[num_vacancies_++] = transition.initial_shell;
        }
        else
        {
            // If there is an Auger subshell, this is a non-radiative
            // transition: create an Auger electron
            CELER_ASSERT(count < secondaries_.size());
            secondaries_[count].particle_id = shared_.electron_id;
            secondaries_[count].direction   = sample_direction_(rng);
            secondaries_[count++].energy    = MevEnergy{transition.energy};

            // Push the new vacancies onto the stack
            CELER_ASSERT(num_vacancies_ + 1 < vacancies_.size());
            vacancies_[num_vacancies_++] = transition.initial_shell;
            vacancies_[num_vacancies_++] = transition.auger_shell;
        }
    }

    return count;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
