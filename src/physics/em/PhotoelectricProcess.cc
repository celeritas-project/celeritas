//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhotoelectricProcess.cc
//---------------------------------------------------------------------------//
#include "PhotoelectricProcess.hh"

#include <utility>
#include "LivermorePEModel.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from host data.
 */
PhotoelectricProcess::PhotoelectricProcess(SPConstParticles particles,
                                           SPConstImported  process_data,
                                           SPConstData      data)
    : particles_(std::move(particles))
    , imported_(process_data,
                particles_,
                ImportProcessClass::photoelectric,
                {pdg::gamma()})
    , data_(std::move(data))
{
    CELER_EXPECT(particles_);
    CELER_EXPECT(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with atomic relaxation data.
 */
PhotoelectricProcess::PhotoelectricProcess(SPConstParticles   particles,
                                           SPConstImported    process_data,
                                           SPConstData        data,
                                           SPConstAtomicRelax atomic_relaxation,
                                           size_type vacancy_stack_size)
    : PhotoelectricProcess(
        std::move(particles), std::move(process_data), std::move(data))
{
    atomic_relaxation_  = std::move(atomic_relaxation);
    vacancy_stack_size_ = vacancy_stack_size;
    CELER_ENSURE(atomic_relaxation_);
    CELER_ENSURE(vacancy_stack_size_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the models associated with this process.
 */
auto PhotoelectricProcess::build_models(ModelIdGenerator next_id) const
    -> VecModel
{
    if (atomic_relaxation_)
    {
        // Construct model with atomic relaxation enabled
        return {std::make_shared<LivermorePEModel>(next_id(),
                                                   *particles_,
                                                   *data_,
                                                   *atomic_relaxation_,
                                                   vacancy_stack_size_)};
    }
    else
    {
        // Construct model without atomic relaxation
        return {std::make_shared<LivermorePEModel>(
            next_id(), *particles_, *data_)};
    }
}

//---------------------------------------------------------------------------//
/*!
 * Get the interaction cross sections for the given energy range.
 */
auto PhotoelectricProcess::step_limits(Applicability applic) const
    -> StepLimitBuilders
{
    return imported_.step_limits(std::move(applic));
}

//---------------------------------------------------------------------------//
/*!
 * Name of the process.
 */
std::string PhotoelectricProcess::label() const
{
    return "Photoelectric effect";
}

//---------------------------------------------------------------------------//
} // namespace celeritas
