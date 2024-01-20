//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/ExceptionHandler.hh
//---------------------------------------------------------------------------//
#pragma once

#include <exception>
#include <functional>
#include <G4ExceptionSeverity.hh>
#include <G4StateManager.hh>
#include <G4Types.hh>
#include <G4VExceptionHandler.hh>

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Abort the event or run in case of an error.
 */
class ExceptionHandler : public G4VExceptionHandler
{
  public:
    //!@{
    //! \name Type aliases
    using StdExceptionHandler = std::function<void(std::exception_ptr)>;
    //!@}

  public:
    explicit ExceptionHandler(StdExceptionHandler handle_exception);

    // Accept error codes from geant4
    G4bool Notify(char const* originOfException,
                  char const* exceptionCode,
                  G4ExceptionSeverity severity,
                  char const* description) final;

  private:
    StdExceptionHandler handle_;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
