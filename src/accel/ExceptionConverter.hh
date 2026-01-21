//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/ExceptionConverter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <exception>

namespace celeritas
{
class SharedParams;
//---------------------------------------------------------------------------//
/*!
 * Translate Celeritas C++ exceptions into Geant4 G4Exception calls.
 *
 * This should generally be used when wrapping calls to Celeritas in a user
 * application.
 *
 * For example, the user event action to transport particles on device could be
 * used as:
 * \code
   void EventAction::EndOfEventAction(const G4Event*)
   {
       // Transport any tracks left in the buffer
       celeritas::ExceptionConverter call_g4exception{"celer.event.flush"};
       CELER_TRY_HANDLE(transport_->Flush(), call_g4exception);

       // Debug error checking
       CELER_ENSURE(transport_.GetBufferSize() == 0
                    || call_g4exception.forwarded());
   }
 * \endcode
 */
class ExceptionConverter
{
  public:
    // Construct with "error code" and optional pointer to shared params
    inline ExceptionConverter(char const* err_code, SharedParams const* params);

    // Construct with just an "error code"
    inline explicit ExceptionConverter(char const* err_code);

    // Capture the current exception and convert it to a G4Exception call
    void operator()(std::exception_ptr p);

    //! Whether an exception was passed to G4Exception
    bool forwarded() const { return forwarded_; }

  private:
    char const* err_code_;
    SharedParams const* params_{nullptr};
    bool forwarded_{false};

    void convert_device_exceptions(std::exception_ptr p) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with an error code and shared parameters.
 *
 * The error code is reported to the Geant4 exception manager. The shared
 * parameters are used to translate internal particle data if an exception
 * occurs.
 */
ExceptionConverter::ExceptionConverter(char const* err_code,
                                       SharedParams const* params)
    : err_code_{err_code}, params_(params)
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct with an error code for dispatching to Geant4.
 */
ExceptionConverter::ExceptionConverter(char const* err_code)
    : ExceptionConverter(err_code, nullptr)
{
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
