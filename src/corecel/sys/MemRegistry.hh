//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/MemRegistry.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/math/Quantity.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! SI prefix for multiples of 1024
struct Kibi
{
    using value_type = std::size_t;

    static CELER_CONSTEXPR_FUNCTION value_type value() { return 1024u; }
    static char const* label() { return "kibi"; }
};

//! 1024 bytes
using KibiBytes = Quantity<Kibi, Kibi::value_type>;
//! Ordered identifiers for memory allocation segments
using MemUsageId = OpaqueId<struct MemUsageEntry>;

//! Statistics about a block of memory usage
struct MemUsageEntry
{
    //! Name of this entry
    std::string label;
    //! Index of the umbrella entry
    MemUsageId parent_index{};
    //! Difference in CPU memory usage from beginning to end
    KibiBytes cpu_delta{};
    //! Reported CPU "high water mark" at the end the block
    KibiBytes cpu_hwm{};
    //! Difference in GPU memory usage from beginning to end
    KibiBytes gpu_delta{};
    //! Reported GPU "high water mark" at the end the block
    KibiBytes gpu_usage{};
};

//---------------------------------------------------------------------------//
/*!
 * Track memory usage across the application.
 *
 * This class is \em not thread-safe and should generally be used during setup.
 * The memory usage entries are a tree. Pushing and popping should be done with
 * \c ScopedMem .
 */
class MemRegistry
{
  public:
    // Construct with no entries
    MemRegistry() = default;

    //// ACCESSORS ////

    //! Number of entries
    MemUsageId::size_type size() const { return entries_.size(); }

    // Get the entry for an ID
    inline MemUsageEntry& get(MemUsageId id);

    // Get the entry for an ID
    inline MemUsageEntry const& get(MemUsageId id) const;

    //// MUTATORS ////

    // Create a new entry and push it onto the stack, returning the new ID
    MemUsageId push();

    // Pop the last entry
    void pop();

  private:
    std::vector<MemUsageEntry> entries_;
    std::vector<MemUsageId> stack_;
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
// Globally shared registry of memory usage
MemRegistry& mem_registry();

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get the entry for an ID.
 */
MemUsageEntry const& MemRegistry::get(MemUsageId id) const
{
    CELER_EXPECT(id < this->size());
    return entries_[id.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * Get the entry for an ID.
 */
MemUsageEntry& MemRegistry::get(MemUsageId id)
{
    CELER_EXPECT(id < this->size());
    return entries_[id.unchecked_get()];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
