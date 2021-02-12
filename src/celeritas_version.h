/*----------------------------------*-C-*------------------------------------*
 * Copyright 2020 UT-Battelle, LLC and other Celeritas Developers.
 * See the top-level COPYRIGHT file for details.
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *---------------------------------------------------------------------------*/
/*! \file celeritas_version.h
 * Version metadata for Celeritas.
 *---------------------------------------------------------------------------*/
#ifndef celeritas_version_h
#define celeritas_version_h

#ifdef __cplusplus
extern "C" {
#endif

extern const char celeritas_version[];
extern const int  celeritas_version_major;
extern const int  celeritas_version_minor;
extern const int  celeritas_version_patch;

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* celeritas_version_h */
