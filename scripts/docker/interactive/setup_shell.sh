#!/bin/bash

# check if CAP_SYS_ADMIN is set
IFS=","
read -ra capabilities <<< "$(getpcaps $$)"


for cap in "${capabilities[@]}"
do
  if [[ "${cap}" == "cap_sys_admin" ]]
  then
    has_sys_admin_cap=
    break
  fi
done

# mount tracefs

if [[ -n ${has_sys_admin_cap+x} ]]
then
  if [[ ! -f /sys/kernel/tracing/current_tracer ]]; then
    mount -t nodev tracefs /sys/kernel/tracing
  fi
else
  echo "cap_sys_admin not set, can't use tracefs-based tracing, run the container with '--cap-add=SYS_ADMIN'"
fi

# activate spack env
if [[ -t 0 ]]
then
  spack env activate -p "${CELERITAS_ENV_NAME}"
fi
