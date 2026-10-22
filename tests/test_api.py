from __future__ import annotations

import time

from fastapi.testclient import TestClient
import pytest

from app import config, main
from tests.fixtures import make_video


@pytest.fixture
def client(tmp_path,monkeypatch):
    root=tmp_path/'isolated';root.mkdir()
    for name,value in [('ROOT',root),('CACHE',root/'cache'),('RUNTIME',root/'.runtime'),('LOGS',root/'logs')]:
        monkeypatch.setattr(config,name,value)
    with TestClient(main.app) as test_client:
        yield test_client
    assert not any(p.is_file() for p in (root/'cache').rglob('*'))


def wait_done(client,job_id):
    deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        result=client.get('/api/jobs/'+job_id).json()
        if result['status'] in {'done','partial','failed','cancelled'}:return result
        time.sleep(.03)
    pytest.fail('processing timeout')


def test_batch_validation_download_range_and_cleanup(client,tmp_path):
    source=tmp_path/'input.mp4';make_video(source,vfr=True)
    response=client.post('/api/uploads',files=[('files',('视频.mp4',source.read_bytes(),'video/mp4')),('files',('bad.mp4',b'bad video','video/mp4'))])
    assert response.status_code==201,response.text
    batch=response.json();job_id=batch['id']
    assert batch['files'][0]['metadata']['width']==160
    assert batch['files'][1]['status']=='failed'
    preview=client.post('/api/plan',json={'upload_id':job_id,'target':{'preset':'custom','edge':'long','pixels':240}})
    assert preview.json()['files'][0]['target']['width']==240
    response=client.post('/api/jobs',json={'upload_id':job_id,'options':{'model':'lanczos','target':{'preset':'custom','edge':'long','pixels':240}}})
    assert response.status_code==202,response.text
    assert client.post('/api/jobs',json={'upload_id':job_id}).status_code==400
    result=wait_done(client,job_id)
    assert result['status']=='partial',result
    assert result['files'][0]['evidence']['frames']==6
    download=result['files'][0]['download_url']
    response=client.get(download)
    assert response.status_code==200 and response.content[4:8]==b'ftyp'
    assert "filename*=utf-8''" in response.headers['content-disposition']
    response=client.get(download,headers={'Range':'bytes=0-99'})
    assert response.status_code==206 and len(response.content)==100
    assert client.get(download,headers={'Range':'bytes=99999999-'}).status_code==416
    assert not any((config.CACHE/'uploads'/job_id).iterdir())
    assert client.delete('/api/jobs/'+job_id).json()['status']=='deleted'
    assert client.get('/api/jobs/'+job_id).status_code==404
    assert not (config.CACHE/'results'/job_id).exists()


def test_invalid_and_cross_origin(client):
    assert client.post('/api/uploads',files={'files':('bad.mp4',b'not a video')}).status_code==400
    assert not list((config.CACHE/'uploads').iterdir())
    assert client.post('/api/plan',json={'upload_id':'missing','target':{'preset':'custom','pixels':-1}}).status_code==422
    assert client.post('/api/jobs',json={'upload_id':'missing'},headers={'Origin':'https://unrelated.example'}).status_code==403
    assert client.post('/api/admin/shutdown').status_code==403


def test_uploaded_cancel_and_expiry(client,tmp_path):
    source=tmp_path/'input.mp4';make_video(source,audio=False)
    def upload():return client.post('/api/uploads',files={'files':('in.mp4',source.read_bytes())}).json()['id']
    cancelled=upload()
    response=client.post('/api/jobs/'+cancelled+'/cancel')
    assert response.json()['status']=='cancelled'
    assert not (config.CACHE/'uploads'/cancelled).exists()
    expired=upload()
    main.manager.records[expired]['updated_at']=0
    main.manager.cleanup_expired()
    assert client.get('/api/jobs/'+expired).status_code==404
    assert not (config.CACHE/'uploads'/expired).exists()
