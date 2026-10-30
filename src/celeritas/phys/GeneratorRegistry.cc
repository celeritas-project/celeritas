//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/GeneratorRegistry.cc
//---------------------------------------------------------------------------//
#include "GeneratorRegistry.hh"

#include "corecel/cont/Range.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Register generators.
 */
void GeneratorRegistry::insert(SPGenerator generator)
{
    auto label = std::string{generator->label()};
    CELER_VALIDATE(!label.empty(), << "generator label is empty");

    auto id = generator->generator_id();
    CELER_VALIDATE(id == this->next_id(),
                   << "incorrect id {" << id.unchecked_get()
                   << "} for generator '" << label << "' (should be {"
                   << this->next_id().get() << "})");

    auto iter_inserted = gen_ids_.insert({label, id});
    CELER_VALIDATE(iter_inserted.second,
                   << "duplicate generator label '" << label << "'");

    generators_.push_back(std::move(generator));
    labels_.push_back(std::move(label));

    CELER_ENSURE(gen_ids_.size() == generators_.size());
    CELER_ENSURE(labels_.size() == generators_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Find the generator corresponding to a label.
 */
GeneratorId GeneratorRegistry::find(std::string const& label) const
{
    auto iter = gen_ids_.find(label);
    if (iter == gen_ids_.end())
        return {};
    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Update the host copy of the generator counters.
 *
 * This waits for the generator counters to be copied from the device, so it
 * should not be called while stepping except for diagnostics.
 */
void GeneratorRegistry::sync_counters(AuxStateVec& aux) const
{
    for (auto id : range(GeneratorId{this->size()}))
    {
        this->at(id)->counters(aux).sync_counters();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Clear the generated counts at the start of a transport loop.
 */
void GeneratorRegistry::clear_generated(AuxStateVec& aux) const
{
    for (auto id : range(GeneratorId{this->size()}))
    {
        this->at(id)->counters(aux).clear_generated();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Reset the generators if the loop aborted early.
 */
void GeneratorRegistry::reset(AuxStateVec& aux) const
{
    for (auto id : range(GeneratorId{this->size()}))
    {
        this->at(id)->reset(aux);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
