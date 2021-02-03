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
CELER_FUNCTION size_type AtomicRelaxation::operator()(size_type shell_id,
                                                      Engine&   rng)
{
    // The EADL only provides transition probabilities for 6 <= Z <= 100, so
    // atomic relaxation is not applicable for Z < 6. Also, transitions are
    // only provided for K, L, M, N, and some O shells.
    const AtomicRelaxElement& el = shared_.elements[el_id_.get()];
    if (!el || shell_id >= el.shells.size())
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
        size_type vacancy_id = vacancies_[--num_vacancies_];

        // If there are no transitions, process the next vacancy
        if (vacancy_id == shared_.unassigned)
        {
            continue;
        }

        // Sample the transition
        const AtomicRelaxSubshell& shell = el.shells[vacancy_id];
        size_type                  tr_id;
        real_type                  prob = generate_canonical(rng);
        for (tr_id = 0; tr_id < shell.transition_prob.size(); ++tr_id)
        {
            if ((prob -= shell.transition_prob[tr_id]) <= 0)
            {
                break;
            }
        }

        // Get the Auger electron shell ID: it will be unassigned if simulation
        // of non-radiative transitions is disabled or if a radiative
        // transition was sampled
        size_type auger_id = shell.auger_shell.empty()
                                 ? shared_.unassigned
                                 : shell.auger_shell[tr_id];

        if (auger_id == shared_.unassigned)
        {
            // If no Auger subshell is provided, this is a radiative
            // transition: create a fluorescence photon
            CELER_ASSERT(count < secondaries_.size());
            secondaries_[count].particle_id = shared_.gamma_id;
            secondaries_[count].direction   = sample_direction_(rng);
            secondaries_[count++].energy
                = MevEnergy{shell.transition_energy[tr_id]};

            // Push the new vacancy onto the stack
            CELER_ASSERT(num_vacancies_ < vacancies_.size());
            vacancies_[num_vacancies_++] = shell.initial_shell[tr_id];
        }
        else
        {
            // If there is an Auger subshell, this is a non-radiative
            // transition: create an Auger electron
            CELER_ASSERT(count < secondaries_.size());
            secondaries_[count].particle_id = shared_.electron_id;
            secondaries_[count].direction   = sample_direction_(rng);
            secondaries_[count++].energy
                = MevEnergy{shell.transition_energy[tr_id]};

            // Push the new vacancies onto the stack
            CELER_ASSERT(num_vacancies_ + 1 < vacancies_.size());
            vacancies_[num_vacancies_++] = shell.initial_shell[tr_id];
            vacancies_[num_vacancies_++] = auger_id;
        }
    }

    return count;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
