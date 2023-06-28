//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/Interface.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
class Interface
{
  protected:
    //!@{
    //! Prevent slicing by deleting move/copy operations
    Interface() = default;
    Interface(Interface const&) = delete;
    Interface& operator=(Interface const&) = delete;
    Interface(Interface&&) = delete;
    Interface& operator=(Interface&&) = delete;
    //!@}
};

//---------------------------------------------------------------------------//
}  // namespace celeritas