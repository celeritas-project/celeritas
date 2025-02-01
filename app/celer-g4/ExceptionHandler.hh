//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/ExceptionHandler.hh
//---------------------------------------------------------------------------//
#pragma once

#include <exception>
#include <functional>
#include <memory>
#include <G4ExceptionSeverity.hh>
#include <G4StateManager.hh>
#include <G4Types.hh>
#include <G4VExceptionHandler.hh>

namespace celeritas
{
class SharedParams;

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
    using SPConstParams = std::shared_ptr<SharedParams const>;
    //!@}

  public:
    ExceptionHandler(StdExceptionHandler handle_exception,
                     SPConstParams params);

    // Accept error codes from geant4
    G4bool Notify(char const* originOfException,
                  char const* exceptionCode,
                  G4ExceptionSeverity severity,
                  char const* description) final;

  private:
    StdExceptionHandler handle_;
    SPConstParams params_;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
