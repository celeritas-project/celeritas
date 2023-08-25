//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/EventIOInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Macros.hh"

namespace celeritas
{
struct Primary;
//---------------------------------------------------------------------------//
/*!
 * Abstract base class for writing all primaries from an event.
 */
class EventWriterInterface
{
  public:
    //!@{
    //! \name Type aliases
    using VecPrimary = std::vector<Primary>;
    using argument_type = VecPrimary const&;
    //!@}

  public:
    virtual ~EventWriterInterface() = default;

    //! Write all primaries from a single event
    virtual void operator()(argument_type primaries) = 0;

  protected:
    //!@{
    //! Allow construction and assignment only through daughter classes
    EventWriterInterface() = default;
    CELER_DEFAULT_COPY_MOVE(EventWriterInterface);
    //!@}
};

//---------------------------------------------------------------------------//
/*!
 * Abstract base class for reading all primaries from an event.
 */
class EventReaderInterface
{
  public:
    //!@{
    //! \name Type aliases
    using VecPrimary = std::vector<Primary>;
    using result_type = VecPrimary;
    //!@}

  public:
    virtual ~EventReaderInterface() = default;

    //! Read all primaries from a single event
    virtual result_type operator()() = 0;

  protected:
    //!@{
    //! Allow construction and assignment only through daughter classes
    EventReaderInterface() = default;
    CELER_DEFAULT_COPY_MOVE(EventReaderInterface);
    //!@}
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
