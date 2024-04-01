//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalState.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
class OpticalParams;
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    OpticalState ...;
   \endcode
 */
class OpticalStateInterface
{
  public:
    //!@{
    //! \name Type aliases
    //!@}
  public:
  protected:
    ~OpticalStateInterface() = default;
};

template<MemSpace M>
class OpticalState final : public OpticalStateInterface
{
  public:
    //!@{
    //! \name Type aliases
    //!@}
  public:
    OpticalState(OpticalParams const& params);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
OpticalState::OpticalState() {}

//---------------------------------------------------------------------------//
}  // namespace celeritas
