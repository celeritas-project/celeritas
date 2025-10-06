//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/GeneratorInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string_view>

#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/data/AuxInterface.hh"
#include "corecel/data/AuxStateVec.hh"

#include "GeneratorCounters.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// TYPES
//---------------------------------------------------------------------------//

//! Index of generator
using GeneratorId = OpaqueId<struct Generator_>;

struct GeneratorStateBase;

//---------------------------------------------------------------------------//
// INTERFACES
//---------------------------------------------------------------------------//
/*!
 * Interface class for generating new tracks.
 *
 * Generators store information about pending primary or secondary particles as
 * auxiliary data and initialize the new tracks directly in the vacant slots.
 */
class GeneratorInterface
{
  public:
    // Anchored virtual destructor for polymorphism
    virtual ~GeneratorInterface();

    //! Index of this class instance in its registry
    virtual GeneratorId generator_id() const = 0;

    //! Short unique label of the generator
    virtual std::string_view label() const = 0;

    //! Get generator counters (mutable)
    virtual GeneratorStateBase& counters(AuxStateVec&) const = 0;

    //! Get generator counters
    virtual GeneratorStateBase const& counters(AuxStateVec const&) const = 0;

  protected:
    //!@{
    //! Allow construction and assignment only through daughter classes
    GeneratorInterface() = default;
    CELER_DEFAULT_COPY_MOVE(GeneratorInterface);
    //!@}
};

//---------------------------------------------------------------------------//
/*!
 * Manage counters for generation states.
 */
struct GeneratorStateBase : public AuxStateInterface
{
    //! Counts since the start of the optical loop
    GeneratorCounters<size_type> counters;
    //! Counts accumulated over the event for diagnostics
    GeneratorCounters<std::size_t> accum;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
