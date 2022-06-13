# GT4SD server upload

## Requirements

* docker
* minio
* gt4sd [requirements](https://github.com/GT4SD/gt4sd-core/blob/main/requirements.txt)
-----

## Run a local minio server

If you want to upload your trained models on a local (or cloud) server running `minio` you can:

### 1) Set environment variables

copy exports in `~/.bashrc` or `~/.zshrc`

```sh
export UPDATE_SECRET_KEY=''
export UPDATE_ACCESS_KEY=''
export UPDATE_BUCKET_MODELS='gt4sd-algorithms-models'
export UPDATE_BUCKET_DATA='gt4sd-algorithms-data'
export UPDATE_BUCKET_ARTIFACTS='gt4sd-algorithms-artifacts'
```

### 2) Create a docker container with a minio server

```sh
cd ~/
mkdir localhost-server
cd localhost-server
mkdir env/
echo >> docker-compose.yml
```

Copy this configuration script in `docker-compose.yml`:

```sh
version: '10'
services:
  cos:
    image: minio/minio:RELEASE.2022-06-07T00-33-41Z
    ports:
      - 9000:9000
    env_file:
     - env/.env.dev 
    environment:
      MINIO_ACCESS_KEY: "${UPDATE_ACCESS_KEY}"
      MINIO_SECRET_KEY: "${UPDATE_SECRET_KEY}"
    command: server /export
  createbuckets:
    image: minio/mc
    depends_on:
      - cos
    env_file:
     - env/.env.dev
    # ensure there is a file in the artifacts bucket
    entrypoint: >
      /bin/sh -c "
      /usr/bin/mc config host add myminio http://cos:9000 ${UPDATE_ACCESS_KEY} ${UPDATE_SECRET_KEY};
      /usr/bin/mc mb myminio/${UPDATE_BUCKET_ARTIFACTS};
      /usr/bin/mc mb myminio/${UPDATE_BUCKET_DATA};
      /usr/bin/mc mb myminio/${UPDATE_BUCKET_MODELS};
      echo 'this is an artifact' >> a_file.txt;
      /usr/bin/mc cp a_file.txt myminio/${UPDATE_BUCKET_ARTIFACTS}/a_file.txt;
      exit 0;
      "
```

You can store default environment variables in `.env.dev`.


### 3) minio configuration

Add the new server to the minio configuration file (`~/.mc/config.json`):

```sh
{
	"version": "10",
	"aliases": {
                "local-dev": {
                        "url": "http://127.0.0.1:9000",
                        "accessKey": "",
                        "secretKey": "",
                        "api": "s3v4",
                        "path": "auto"
                },
                ...
                }
}
```

### 4) run `docker compose up`

After running `docker compose up` inside localhost-server this script will create a local minio server and the bucket `myminio` structure .
If everything is working you should be able to see `a_file.txt` running:

```
mc ls local-dev/gt4sd-algorithms-artifacts/
```
-------

## Upload models

You can now upload your model on the server:

```sh
gt4sd-upload --training_pipeline_name paccmann-vae-trainer --model_path /tmp/gt4sd-paccmann-gp --training_name fast-example --target_version fast-example-v0 --algorithm_application PaccMannGPGenerator
```

You should be able to see this model running:

```sh
mc ls local-dev/gt4sd-algorithms-artifacts/controlled_sampling/PaccMannGP/PaccMannGPGenerator/fast-example-v0/
```
