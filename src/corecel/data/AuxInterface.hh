//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/AuxInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string_view>

#include "corecel/Macros.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/sys/ThreadId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// TYPES
//---------------------------------------------------------------------------//

//! Index for auxiliary data
using AuxId = OpaqueId<struct Aux_>;

class AuxStateInterface;

//---------------------------------------------------------------------------//
// INTERFACES
//---------------------------------------------------------------------------//
/*!
 * Base class for extensible shared data that has associated state.
 *
 * Auxiliary data can be added to a \c AuxParamsInterface at runtime to be
 * passed among multiple classes, and then \c dynamic_cast to the expected
 * type. It needs to supply a factory function for creating the a state
 * instance for multithreaded data on a particular stream and a given memory
 * space. Classes can inherit both from \c AuxParamsInterface and other \c
 * ActionInterface classes.
 */
class AuxParamsInterface
{
  public:
    //@{
    //! \name Type aliases
    using UPState = std::unique_ptr<AuxStateInterface>;
    //@}

  public:
    // Anchored virtual destructor for polymorphism
    virtual ~AuxParamsInterface();

    //! Index of this class instance in its registry
    virtual AuxId aux_id() const = 0;

    //! Label for the auxiliary data
    virtual std::string_view label() const = 0;

    //! Factory function for building multithread state for a stream
    virtual UPState create_state(MemSpace, StreamId, size_type size) const = 0;

  protected:
    //!@{
    //! Allow construction and assignment only through daughter classes
    AuxParamsInterface() = default;
    CELER_DEFAULT_COPY_MOVE(AuxParamsInterface);
    //!@}
};

//---------------------------------------------------------------------------//
/*!
 * Auxiliary state data owned by a single stream.
 *
 * This interface class is strictly to allow polymorphism and dynamic casting.
 */
class AuxStateInterface
{
  public:
    //@{
    //! \name Type aliases
    using SPState = std::shared_ptr<AuxStateInterface>;
    //@}

  public:
    // Anchored virtual destructor for polymorphism
    virtual ~AuxStateInterface();

  protected:
    //!@{
    //! Allow construction and assignment only through daughter classes
    AuxStateInterface() = default;
    CELER_DEFAULT_COPY_MOVE(AuxStateInterface);
    //!@}
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
