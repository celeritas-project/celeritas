//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Material.mocl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Mock (single) material class.
 * 
 * This is a temporary, extremely bare-bones mock class, used currently only
 * for Bethe-Heitler model. To be replaced.
 */
class MaterialMock
{
  public:
    inline CELER_FUNCTION
    MaterialMock(const std::string name,
                 size_type Z,
                 real_type A) :
                 name_(name),
                 Z_(Z),
                 A_(A) {}

    //@{
    //! Accessors
    inline CELER_FUNCTION size_type Z() const { return Z_; };
    inline CELER_FUNCTION real_type A() const { return A_; };
    //@}

  private:
    std::string name_;  //!< Material name
    size_type Z_;       //!< Atomic number
    real_type A_;       //!< Atomic mass
};



//---------------------------------------------------------------------------//
} // namespace celeritas
