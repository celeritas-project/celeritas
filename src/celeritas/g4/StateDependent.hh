//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/g4/StateDependent.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <G4VStateDependent.hh>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 *
 */
class StateDependent : public G4VStateDependent
{
  public:
    //!@{
    //! \name Type aliases
    using AppState = G4ApplicationState;
    using LocalStateChangeFunc
        = std::function<void(StreamId, AppState, AppState)>;

  public:
    StateDependent(StreamId, LocalStateChangeFunc cb);

    G4bool Notify(G4ApplicationState state);

  private:
    StreamId local_stream_;
    LocalStateChangeFunc cb_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
