//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/GeneratorRegistry.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "corecel/data/AuxStateVec.hh"

#include "GeneratorInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Manage classes that generate tracks.
 *
 * This class keeps track of \c GeneratorInterface classes.
 */
class GeneratorRegistry
{
  public:
    //!@{
    //! \name Type aliases
    using SPGenerator = std::shared_ptr<GeneratorInterface>;
    using SPConstGenerator = std::shared_ptr<GeneratorInterface const>;
    //!@}

  public:
    // Default constructor
    GeneratorRegistry() = default;

    //// CONSTRUCTION ////

    //! Get the next available ID
    GeneratorId next_id() const { return GeneratorId(generators_.size()); }

    // Register generator
    void insert(SPGenerator generator);

    //// ACCESSORS ////

    //! Get the number of defined generators
    GeneratorId::size_type size() const { return generators_.size(); }

    //! Whether any generators have been registered
    bool empty() const { return generators_.empty(); }

    // Access generator at the given ID
    inline SPGenerator const& at(GeneratorId);
    inline SPConstGenerator at(GeneratorId) const;

    // Get the label corresponding to the generator
    inline std::string const& id_to_label(GeneratorId id) const;

    // Find the ID corresponding to an label
    GeneratorId find(std::string const& label) const;

    // Reset the generator counters if the loop aborted early
    void reset(AuxStateVec&) const;

  private:
    std::vector<SPGenerator> generators_;
    std::vector<std::string> labels_;
    std::unordered_map<std::string, GeneratorId> gen_ids_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Access mutable generator at the given ID.
 */
auto GeneratorRegistry::at(GeneratorId id) -> SPGenerator const&
{
    CELER_EXPECT(id < this->size());
    return generators_[id.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * Access generator at the given ID.
 */
auto GeneratorRegistry::at(GeneratorId id) const -> SPConstGenerator
{
    CELER_EXPECT(id < this->size());
    return generators_[id.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * Get the label corresponding to the generator.
 */
std::string const& GeneratorRegistry::id_to_label(GeneratorId id) const
{
    CELER_EXPECT(id < this->size());
    return labels_[id.unchecked_get()];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
