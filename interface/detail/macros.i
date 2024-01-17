//---------------------------------*-SWIG-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file detail/macros.i
//---------------------------------------------------------------------------//

%define %celer_rename_to_cstring(CLASS_LOWER, CLASS)
%rename(CLASS_LOWER ## _to_string) to_cstring(CLASS);
%enddef

